"""Hybrid detector: classical blocks plus secondary-detector blocks it missed.

The classical detector is precise on bubbles and captions; an ML line detector (PP-OCR
DB, bundled with RapidOCR) recovers text over dense artwork where connected components
fuse with hatching. Secondary blocks are added only where no classical block overlaps.
"""

from __future__ import annotations

import cv2
import numpy as np

from manga_ar.detect.base import RgbArray, TextBlock, TextDetector
from manga_ar.detect.geometry import disk
from manga_ar.errors import DetectionError
from manga_ar.logging_setup import get_logger

log = get_logger(__name__)


class HybridDetector:
    name = "hybrid"

    def __init__(self, primary: TextDetector, secondary: TextDetector) -> None:
        self.primary = primary
        self.secondary = secondary

    def detect(self, rgb: RgbArray) -> list[TextBlock]:
        blocks = self.primary.detect(rgb)
        try:
            extra = self.secondary.detect(rgb)
        except DetectionError as exc:
            log.warning("secondary detector failed (%s); using primary blocks only", exc)
            return blocks
        for cand in extra:
            if cand.bbox.area < 150:
                continue
            # A secondary block that swallows a much smaller primary block means the
            # primary found only part of the text (typical over hatching): extend it.
            covered = [
                b
                for b in blocks
                if b.bbox.overlap_ratio(cand.bbox) >= 0.8
                and cand.bbox.area >= 1.5 * b.bbox.area
                and _textured_background(rgb, b)
            ]
            if covered:
                for b in covered:
                    blocks.remove(b)
                blocks.append(_merge(cand, covered))
                continue
            clash = any(
                cand.bbox.iou(b.bbox) > 0.1
                or cand.bbox.overlap_ratio(b.bbox) > 0.3
                or b.bbox.overlap_ratio(cand.bbox) > 0.3
                for b in blocks
            )
            if not clash:
                blocks.append(cand)
        blocks.sort(key=lambda b: (b.bbox.y0, b.bbox.x0))
        return blocks


def _textured_background(rgb: RgbArray, block: TextBlock) -> bool:
    """Is the block sitting on artwork (not a bubble/caption interior)?"""
    h, w = rgb.shape[:2]
    pad = max(4, round(0.5 * block.glyph_size))
    box = block.bbox.expand(pad).clip(w, h)
    gray = cv2.cvtColor(rgb[box.y0 : box.y1, box.x0 : box.x1], cv2.COLOR_RGB2GRAY)
    text = (
        block.text_mask.window(box) if block.text_mask is not None else np.zeros(gray.shape, bool)
    )
    background = gray[~(cv2.dilate(text.astype(np.uint8), disk(2)) > 0)]
    return bool(background.size and float(background.std()) > 25.0)


def _merge(cand: TextBlock, covered: list[TextBlock]) -> TextBlock:
    """Secondary block geometry with the union of all glyph pixels."""
    from manga_ar.schemas import CropMask

    mask = cand.text_mask
    for b in covered:
        if b.text_mask is None:
            continue
        if mask is None:
            mask = b.text_mask
            continue
        box = mask.bbox.union(b.text_mask.bbox)
        mask = CropMask(box, mask.window(box) | b.text_mask.window(box))
    cand.text_mask = mask
    if mask is not None:
        cand.bbox = cand.bbox.union(mask.bbox)
    return cand
