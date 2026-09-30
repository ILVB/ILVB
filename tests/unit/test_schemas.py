from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from manga_ar.errors import MangaArError
from manga_ar.schemas import (
    BBox,
    CropMask,
    Flag,
    LayoutResult,
    OcrCandidate,
    OcrResult,
    PageDocument,
    Region,
    RegionOverride,
    RegionType,
    TranslationAttempt,
    TranslationResult,
    rle_decode,
    rle_encode,
)


def test_bbox_geometry() -> None:
    a, b = BBox(0, 0, 10, 10), BBox(5, 5, 15, 15)
    assert a.area == 100 and a.intersection(b) == BBox(5, 5, 10, 10)
    assert a.iou(b) == pytest.approx(25 / 175)
    assert a.union(b) == BBox(0, 0, 15, 15)
    assert BBox(0, 0, 10, 10).contains(BBox(2, 2, 8, 8))
    assert BBox(-5, -5, 50, 50).clip(20, 30) == BBox(0, 0, 20, 30)
    assert a.intersection(BBox(20, 20, 30, 30)) is None and a.iou(BBox(20, 20, 30, 30)) == 0
    with pytest.raises(ValueError):
        BBox(10, 0, 5, 5)


@settings(max_examples=60, deadline=None)
@given(arrays(np.bool_, st.tuples(st.integers(1, 17), st.integers(1, 23))))
def test_rle_roundtrip(mask: np.ndarray) -> None:
    runs = rle_encode(mask)
    assert sum(runs) == mask.size
    assert np.array_equal(rle_decode(runs, mask.shape), mask)


def test_rle_length_mismatch() -> None:
    with pytest.raises(MangaArError):
        rle_decode([3, 2], (2, 2))


def test_cropmask_full_window_translate() -> None:
    full = np.zeros((40, 50), bool)
    full[10:20, 5:30] = True
    full[15, 40] = True
    cm = CropMask.from_full(full)
    assert cm is not None and cm.bbox == BBox(5, 10, 41, 20)
    assert np.array_equal(cm.to_full(40, 50), full)
    win = cm.window(BBox(0, 0, 10, 15))
    assert win.shape == (15, 10) and win[10:, 5:].all() and win.sum() == 25
    assert CropMask.from_full(np.zeros((5, 5), bool)) is None
    moved = cm.translate(3, 4)
    assert moved.bbox == BBox(8, 14, 44, 24) and moved.area == cm.area
    back = CropMask.from_json(cm.to_json())
    assert back.bbox == cm.bbox and np.array_equal(back.data, cm.data)
    assert cm.to_full(12, 12).sum() == full[:12, :12].sum()  # clipped paste


def _doc() -> PageDocument:
    m = np.zeros((6, 8), bool)
    m[1:5, 2:7] = True
    region = Region(
        id="p0-r0",
        type=RegionType.BUBBLE,
        bbox=BBox(10, 10, 60, 40),
        lines=[BBox(10, 10, 60, 20)],
        polygon=[(1, 2), (3, 4)],
        bubble_mask=CropMask(BBox(0, 0, 8, 6), m),
        safe_box=BBox(12, 12, 58, 38),
        source_lang="ja",
        vertical=True,
        fill_color=(250, 250, 250),
        ocr=OcrResult(
            "easyocr",
            "今日は",
            0.9,
            "ja",
            True,
            "今日は♡",
            ["♡"],
            [OcrCandidate("rapid", "今日", 0.4, 0.2)],
        ),
        translation=TranslationResult(
            "tm", "اليوم", False, [TranslationAttempt("google", False, "timeout", 25.0)]
        ),
        layout=LayoutResult(
            "NotoNaskhArabic",
            22,
            ["اليوم"],
            "shape",
            BBox(12, 12, 58, 38),
            28,
            (0, 0, 0),
            (255, 255, 255),
            2,
            ["spacing"],
        ),
        flags={Flag.MERGED, Flag.LOW_CONFIDENCE},
        override=RegionOverride(text="مرحبا", size_px=18),
    )
    return PageDocument(
        source="ch1/page01.png",
        width=100,
        height=80,
        regions=[region],
        config_hash="abc",
        status="ok",
        timings={"ocr": 0.5},
    )


def test_page_document_roundtrip(tmp_path: object) -> None:
    doc = _doc()
    text = doc.dumps()
    assert "مرحبا" in text  # UTF-8, not \u-escaped
    back = PageDocument.from_json(__import__("json").loads(text))
    assert back.to_json() == doc.to_json()
    r = back.regions[0]
    assert r.arabic_text == "مرحبا" and r.flags == {Flag.MERGED, Flag.LOW_CONFIDENCE}
    assert np.array_equal(r.bubble_mask.data, doc.regions[0].bubble_mask.data)  # type: ignore[union-attr]


def test_page_document_save_load(tmp_path: object) -> None:
    from pathlib import Path

    path = Path(str(tmp_path)) / "x.mangaar.json"
    doc = _doc()
    doc.save(path)
    assert PageDocument.load(path).to_json() == doc.to_json()
    path.write_text('{"schema_version": 99}', encoding="utf-8")
    with pytest.raises(MangaArError, match="schema_version"):
        PageDocument.load(path)


def test_arabic_text_precedence() -> None:
    r = Region(id="r", type=RegionType.BUBBLE, bbox=BBox(0, 0, 1, 1))
    assert r.arabic_text is None and not r.is_translated
    r.translation = TranslationResult("google", "نص")
    assert r.arabic_text == "نص"
    r.flag(Flag.UNTRANSLATED)
    assert r.arabic_text is None
    r.override.text = "تعديل"
    assert r.arabic_text == "تعديل"
    r.override.skip = True
    assert not r.is_translated
