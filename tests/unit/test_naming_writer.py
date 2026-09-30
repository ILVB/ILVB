from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from manga_ar.io import writer
from manga_ar.io.naming import (
    is_archive_name,
    is_image_name,
    natural_sorted,
    output_name,
    sidecar_name,
)
from manga_ar.io.writer import atomic_write_bytes, encode_image
from tests.fixtures.synth import gradient_rgb


def test_natural_sort() -> None:
    names = [
        "page10.png",
        "page2.png",
        "Page1.png",
        "page2a.png",
        "ch2/p1.png",
        "ch10/p1.png",
        "ch1/p10.png",
        "ch1/p9.png",
    ]
    assert natural_sorted(names) == [
        "ch1/p9.png",
        "ch1/p10.png",
        "ch2/p1.png",
        "ch10/p1.png",
        "Page1.png",
        "page2.png",
        "page2a.png",
        "page10.png",
    ]


def test_names() -> None:
    assert output_name("p01", "png") == "p01_ar.png"
    assert output_name("p01", "jpeg") == "p01_ar.jpg"
    assert sidecar_name("p01") == "p01_ar.mangaar.json"
    assert is_image_name("A/B.WEBP") and not is_image_name("x.txt")
    assert is_archive_name("vol.CBZ") and not is_archive_name("a.png")


def test_atomic_write_no_partial_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "out.png"
    target.write_bytes(b"old")

    def boom(src: object, dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(writer.Path, "replace", boom)
    with pytest.raises(OSError):
        atomic_write_bytes(target, b"new-data")
    assert target.read_bytes() == b"old"
    assert [p.name for p in tmp_path.iterdir()] == ["out.png"]  # temp file cleaned


def test_atomic_write_creates_dirs(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c.bin"
    atomic_write_bytes(target, b"x")
    assert target.read_bytes() == b"x"
    assert not any(p.name.startswith(".tmp-") for p in target.parent.iterdir())


@pytest.mark.parametrize("fmt", ["png", "jpg", "webp"])
def test_encode_formats(fmt: str, tmp_path: Path) -> None:
    rgb = gradient_rgb()
    data = encode_image(rgb, fmt, jpeg_quality=95)
    p = tmp_path / f"x.{fmt}"
    p.write_bytes(data)
    back = np.asarray(Image.open(p).convert("RGB"))
    assert back.shape == rgb.shape
    if fmt == "png":
        assert np.array_equal(back, rgb)
    else:
        assert np.abs(back.astype(int) - rgb.astype(int)).mean() < 6


def test_grayscale_preserved() -> None:
    gray = np.repeat(np.arange(64, dtype=np.uint8)[None, :, None], 3, axis=2).repeat(8, 0)
    import io

    im = Image.open(io.BytesIO(encode_image(gray, "png", grayscale=True)))
    assert im.mode == "L"
    color = gradient_rgb()
    im2 = Image.open(io.BytesIO(encode_image(color, "png", grayscale=True)))
    assert im2.mode == "RGB"  # colour introduced: stays RGB


def test_encode_rejects_bad_arrays() -> None:
    from manga_ar.errors import MangaArError

    with pytest.raises(MangaArError):
        encode_image(np.zeros((4, 4), np.uint8), "png")
    with pytest.raises(MangaArError):
        encode_image(gradient_rgb(), "gif")
    assert os.sep  # keep import used on all platforms


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
def test_atomic_write_respects_umask(tmp_path: Path) -> None:
    from manga_ar.io.writer import _FILE_MODE, atomic_write_bytes

    target = tmp_path / "out.bin"
    atomic_write_bytes(target, b"x")
    assert target.stat().st_mode & 0o777 == _FILE_MODE
    assert _FILE_MODE & 0o400  # owner can always read
