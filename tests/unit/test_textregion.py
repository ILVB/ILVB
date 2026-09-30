"""1.0.1: TextRegion record, mask references and the legacy Region adapter."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from manga_ar.detect.geometry import ellipse_mask
from manga_ar.schemas import BBox, CropMask, OcrResult, Region, RegionType
from manga_ar.textregion import MaskRef, TextRegion, script_tag
from manga_ar.textregion_legacy import mask_polygon, to_legacy_region, to_text_region


def _mask(box: BBox) -> CropMask:
    data = np.zeros((box.height, box.width), bool)
    data[2:-2, 3:-3] = True
    return CropMask(box, data)


def _region(kind: RegionType = RegionType.BUBBLE) -> Region:
    text_box = BBox(40, 50, 90, 80)
    bubble_box = BBox(20, 30, 120, 110)
    return Region(
        id="p-r1", type=kind, bbox=text_box, text_mask=_mask(text_box),
        bubble_mask=CropMask(bubble_box, ellipse_mask((80, 100), BBox(0, 0, 100, 80))),
        reading_order=2, source_lang="ja", vertical=True,
        ocr=OcrResult(engine="e", text="待って！", confidence=0.9, lang="ja"),
    )  # fmt: skip


@pytest.mark.parametrize(
    ("kind", "text_type"),
    [(RegionType.BUBBLE, "dialogue"), (RegionType.NARRATION, "narration"),
     (RegionType.FREE_TEXT, "sign"), (RegionType.SFX, "sfx")],
)  # fmt: skip
def test_legacy_round_trip(kind: RegionType, text_type: str) -> None:
    tr = to_text_region(_region(kind), "page-1")
    assert tr.type == text_type and tr.page == "page-1" and tr.script == "Jpan"
    assert (tr.text, tr.confidence, tr.lang, tr.reading_order, tr.vertical) == (
        "待って！", 0.9, "ja", 2, True)  # fmt: skip
    assert tr.bbox == BBox(40, 50, 90, 80) and tr.mask is not None
    back = to_legacy_region(tr)
    assert back.type == kind and back.bbox == tr.bbox
    assert back.text_mask is not None
    assert np.array_equal(back.text_mask.to_full(200, 200), _region().text_mask.to_full(200, 200))  # type: ignore[union-attr]
    assert to_text_region(back, "page-1") == tr  # TextRegion → Region → TextRegion is exact


def test_bubble_polygon_reproduces_the_mask() -> None:
    region = _region()
    poly = mask_polygon(region.bubble_mask)  # type: ignore[arg-type]
    assert poly is not None and len(poly) >= 3
    back = to_legacy_region(to_text_region(region, "p")).bubble_mask
    a = region.bubble_mask.to_full(200, 200)  # type: ignore[union-attr]
    b = back.to_full(200, 200)  # type: ignore[union-attr]
    assert (a & b).sum() / (a | b).sum() > 0.97


def test_thought_and_credit_map_to_legacy_kinds() -> None:
    base = to_text_region(_region(), "p")
    assert to_legacy_region(base.model_copy(update={"type": "thought"})).type == RegionType.BUBBLE
    credit = base.model_copy(update={"type": "credit"})
    assert to_legacy_region(credit).type == RegionType.FREE_TEXT


def test_validation() -> None:
    with pytest.raises(ValidationError):
        TextRegion(id="x", page="p", polygon=((0, 0), (1, 1)))  # not a polygon
    with pytest.raises(ValidationError):
        TextRegion(id="x", page="p", polygon=((0, 0), (1, 0), (1, 1)), confidence=1.5)
    with pytest.raises(ValidationError):
        TextRegion(id="x", page="p", polygon=((0, 0), (1, 0), (1, 1)), type="speech")  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="cover"):
        MaskRef(bbox=(0, 0, 4, 4), rle=(3, 2))
    ref = MaskRef.from_crop(_mask(BBox(0, 0, 10, 8)))
    assert MaskRef.model_validate_json(ref.model_dump_json()) == ref


def test_script_tag() -> None:
    assert [script_tag(t) for t in ("待って", "谢谢你", "고마워", "Wait!", "…!?")] == [
        "Jpan", "Hani", "Hang", "Latn", None]  # fmt: skip
