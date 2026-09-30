"""Adapter between v0.1.0 ``schemas.Region`` and the v0.2.0 ``TextRegion`` record.

Type mapping (ADR-0001 #4): bubble ↔ dialogue (thought also maps to bubble), narration ↔
narration, free_text ↔ sign (credit also maps to free_text), sfx ↔ sfx.
"""

from __future__ import annotations

import cv2
import numpy as np

from manga_ar.schemas import BBox, CropMask, OcrResult, Region, RegionType
from manga_ar.textregion import MaskRef, Point, TextRegion, TextType, script_tag

TO_TEXT_TYPE: dict[RegionType, TextType] = {
    RegionType.BUBBLE: "dialogue",
    RegionType.NARRATION: "narration",
    RegionType.FREE_TEXT: "sign",
    RegionType.SFX: "sfx",
}
TO_LEGACY_TYPE: dict[str, RegionType] = {
    "dialogue": RegionType.BUBBLE,
    "thought": RegionType.BUBBLE,
    "narration": RegionType.NARRATION,
    "sign": RegionType.FREE_TEXT,
    "credit": RegionType.FREE_TEXT,
    "sfx": RegionType.SFX,
}


def _box_polygon(box: BBox) -> tuple[Point, ...]:
    return ((box.x0, box.y0), (box.x1, box.y0), (box.x1, box.y1), (box.x0, box.y1))


def mask_polygon(mask: CropMask) -> tuple[Point, ...] | None:
    """Outer contour of the largest component, in page coordinates (pixel corners)."""
    data = np.pad(mask.data.astype(np.uint8), 1)
    contours, _ = cv2.findContours(data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    pts = max(contours, key=cv2.contourArea).reshape(-1, 2)
    if len(pts) < 3:
        return None
    ox, oy = mask.bbox.x0 - 1, mask.bbox.y0 - 1
    return tuple((float(x + ox), float(y + oy)) for x, y in pts)


def polygon_mask(polygon: tuple[Point, ...]) -> CropMask | None:
    pts = np.round(np.asarray(polygon, dtype=np.float64)).astype(np.int32)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0) + 1
    canvas = np.zeros((int(y1 - y0), int(x1 - x0)), np.uint8)
    cv2.fillPoly(canvas, [pts - [x0, y0]], 1)
    return CropMask.from_full(canvas > 0, (int(x0), int(y0)))


def to_text_region(region: Region, page: str) -> TextRegion:
    ocr = region.ocr
    text = ocr.text if ocr is not None else ""
    polygon = tuple((float(x), float(y)) for x, y in region.polygon) or _box_polygon(region.bbox)
    return TextRegion(
        id=region.id,
        page=page,
        polygon=polygon,
        mask=MaskRef.from_crop(region.text_mask) if region.text_mask is not None else None,
        text=text,
        confidence=ocr.confidence if ocr is not None else None,
        script=script_tag(text),
        lang=region.source_lang or (ocr.lang if ocr is not None else None),
        type=TO_TEXT_TYPE[region.type],
        reading_order=region.reading_order,
        bubble_polygon=mask_polygon(region.bubble_mask) if region.bubble_mask else None,
        vertical=region.vertical,
    )


def to_legacy_region(tr: TextRegion) -> Region:
    bubble = polygon_mask(tr.bubble_polygon) if tr.bubble_polygon is not None else None
    ocr = None
    if tr.text or tr.confidence is not None:
        ocr = OcrResult(engine="textregion", text=tr.text, confidence=tr.confidence,
                        lang=tr.lang, vertical=tr.vertical)  # fmt: skip
    return Region(
        id=tr.id,
        type=TO_LEGACY_TYPE[tr.type],
        bbox=tr.bbox,
        polygon=[(round(x), round(y)) for x, y in tr.polygon],
        bubble_mask=bubble,
        text_mask=tr.mask.to_crop() if tr.mask is not None else None,
        reading_order=tr.reading_order,
        source_lang=tr.lang,
        vertical=tr.vertical,
        ocr=ocr,
    )
