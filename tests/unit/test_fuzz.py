"""Fuzzing (Phase 7): loader, archive names, normalisers, Arabic shaping and sidecars.

Properties: hostile input only ever raises the documented typed errors, and pure text
functions never crash, are idempotent where they must be, and keep their invariants.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import zipfile

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from PIL import Image

from manga_ar.errors import ArchiveError, ImageLoadError
from manga_ar.io.archive import read_archive, safe_member_name
from manga_ar.io.loader import load_image_bytes
from manga_ar.schemas import BBox, CropMask, Flag, PageDocument, Region, RegionType
from manga_ar.translate.normalize_ar import BIDI_CONTROLS, ZERO_WIDTH, normalize_ar
from manga_ar.typeset.arabic_text import shape_line, visual_order

# Small by default (unit budget). Deep run:
#   MANGAAR_FUZZ_EXAMPLES=2000 pytest tests/unit/test_fuzz.py
EXAMPLES = int(os.environ.get("MANGAAR_FUZZ_EXAMPLES", "20"))
FUZZ = settings(
    max_examples=EXAMPLES,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)


def _encoded(mode: str, fmt: str, w: int, h: int, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    if mode in {"L", "P"}:
        arr = rng.integers(0, 256, (h, w), dtype=np.uint8)
        img = Image.fromarray(arr, "L")
        img = img.convert("P") if mode == "P" else img
    elif mode == "I;16":
        img = Image.fromarray(rng.integers(0, 65535, (h, w), dtype=np.uint16), "I;16")
    else:
        img = Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), "RGB")
        img = img.convert(mode)
    buf = io.BytesIO()
    img.save(buf, fmt)
    return buf.getvalue()


@FUZZ
@given(
    mode=st.sampled_from(["RGB", "RGBA", "L", "P", "CMYK", "LA", "I;16"]),
    fmt=st.sampled_from(["PNG", "TIFF", "WEBP", "BMP", "JPEG"]),
    seed=st.integers(0, 10_000),
    flips=st.lists(st.tuples(st.integers(0, 10_000), st.integers(0, 255)), max_size=6),
    cut=st.one_of(st.none(), st.integers(0, 4000)),
)
def test_loader_never_raises_untyped(
    mode: str, fmt: str, seed: int, flips: list[tuple[int, int]], cut: int | None
) -> None:
    try:
        encoded = _encoded(mode, fmt, 32, 24, seed)
    except (OSError, ValueError, KeyError):
        encoded = b""
    assume(encoded)  # this Pillow cannot write that mode in that format
    data = bytearray(encoded)
    for pos, value in flips:
        data[pos % len(data)] = value
    if cut is not None:
        data = data[: cut % (len(data) + 1)]
    for policy in ("lenient", "strict"):
        from manga_ar.io.loader import LoadLimits

        try:
            img = load_image_bytes(bytes(data), f"fuzz.{fmt.lower()}", LoadLimits(truncated=policy))
        except ImageLoadError:
            continue
        assert img.rgb.dtype == np.uint8 and img.rgb.ndim == 3 and img.rgb.shape[2] == 3
        assert img.rgb.flags.c_contiguous and min(img.rgb.shape[:2]) >= 16


@FUZZ
@given(st.binary(max_size=512))
def test_loader_random_bytes(data: bytes) -> None:
    with contextlib.suppress(ImageLoadError):
        load_image_bytes(data, "random.png")


@FUZZ
@given(st.text(min_size=1, max_size=60))
def test_archive_member_names(raw: str) -> None:
    try:
        name = safe_member_name(raw)
    except ArchiveError:
        return
    parts = name.split("/")
    assert ".." not in parts and "" not in parts and not name.startswith("/")
    assert "\\" not in name and not (len(name) > 1 and name[1] == ":")


@FUZZ
@given(
    members=st.lists(
        st.tuples(st.text(min_size=1, max_size=20), st.binary(max_size=64)), max_size=6
    )
)
def test_archive_reader_is_typed(
    members: list[tuple[str, bytes]], tmp_path_factory: pytest.TempPathFactory
) -> None:
    path = tmp_path_factory.mktemp("zip") / "fuzz.cbz"
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members:
            try:
                zf.writestr(name, data)
            except (ValueError, UnicodeEncodeError, IndexError):  # zipfile cannot write it
                continue
    try:
        contents = read_archive(path)
    except ArchiveError:
        return
    for m in contents.members:
        assert safe_member_name(m.name) == m.name


_TEXT = st.text(
    alphabet=st.one_of(
        st.characters(min_codepoint=0x0600, max_codepoint=0x06FF),
        st.characters(min_codepoint=0xFE70, max_codepoint=0xFEFF),
        st.sampled_from(list(BIDI_CONTROLS + ZERO_WIDTH + "ـ .,;?!…()[]«»「」0123456789OKab ♡♪")),
        st.characters(codec="utf-8"),
    ),
    max_size=120,
)


@FUZZ
@given(text=_TEXT, digits=st.sampled_from(["western", "arabic_indic"]))
def test_normalize_ar_properties(text: str, digits: str) -> None:
    once = normalize_ar(text, digits)
    assert normalize_ar(once, digits) == once  # idempotent (typesetting re-normalises)
    assert not any(c in BIDI_CONTROLS + ZERO_WIDTH for c in once)
    assert once == once.strip() and "  " not in once
    assert len(normalize_ar(text * 10, digits, max_chars=50)) <= 51


def _mirror_class(ch: str) -> str:
    from bidi.mirror import MIRRORED

    return min(ch, str(MIRRORED.get(ch, ch)))


@FUZZ
@given(_TEXT)
def test_shaping_and_bidi_never_crash(text: str) -> None:
    line = normalize_ar(text)
    visual = shape_line(line)
    assert isinstance(visual, str)
    plain = visual_order(line)
    # reordering + mirroring permutes characters (up to mirror pairs), never adds or drops
    assert sorted(map(_mirror_class, plain)) == sorted(map(_mirror_class, line))


@FUZZ
@given(
    w=st.integers(1, 40),
    h=st.integers(1, 40),
    seed=st.integers(0, 1000),
    text=st.text(max_size=30),
)
def test_sidecar_round_trip(w: int, h: int, seed: int, text: str) -> None:
    rng = np.random.default_rng(seed)
    mask = rng.random((h, w)) > 0.5
    box = BBox(3, 4, 3 + w, 4 + h)
    region = Region(
        id="r", type=RegionType.BUBBLE, bbox=box, bubble_mask=CropMask(box, mask),
        flags={Flag.MERGED},
    )  # fmt: skip
    region.override.text = text or None
    doc = PageDocument(source=text or "p", width=64, height=64, regions=[region])
    back = PageDocument.from_json(json.loads(doc.dumps()))
    assert back.regions[0].bubble_mask is not None
    assert np.array_equal(back.regions[0].bubble_mask.data, mask)
    assert back.regions[0].override.text == region.override.text
    assert back.dumps() == doc.dumps()
