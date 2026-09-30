"""Residual-text check: re-run the detector on the cleaned crop (S4)."""

from __future__ import annotations

from manga_ar.detect.base import TextDetector
from manga_ar.inpaint.base import RgbArray
from manga_ar.schemas import Region


class ResidualChecker:
    def __init__(self, detector: TextDetector) -> None:
        self.detector = detector

    def has_residual(self, page: RgbArray, region: Region) -> bool:
        """True if text is still detected where the region's text was."""
        h, w = page.shape[:2]
        box = region.bbox.expand(12).clip(w, h)
        crop = page[box.y0 : box.y1, box.x0 : box.x1]
        target = region.bbox.translate(-box.x0, -box.y0)
        for block in self.detector.detect(crop):
            if block.is_sfx:
                continue
            if block.bbox.overlap_ratio(target) >= 0.5 and block.bbox.area >= 0.02 * target.area:
                return True
        return False
