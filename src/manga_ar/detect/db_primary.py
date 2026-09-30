"""DB-primary detector (v2): PP-OCR DB blocks, extended by classical blocks that contain them.

PP-OCR DB (bundled with RapidOCR) finds text over screentone and artwork where connected
components fail, and rejects most texture false positives; it sometimes misses the short
last column of a vertical block or a tiny line. A classical block may therefore *extend*
the DB blocks it covers (its geometry, the union of glyph pixels) but never enters on its
own, so classical false positives on texture stay out (ADR-0004).
"""

from __future__ import annotations

from manga_ar.detect.base import RgbArray, TextBlock, TextDetector
from manga_ar.detect.hybrid import _merge
from manga_ar.errors import DetectionError
from manga_ar.logging_setup import get_logger

log = get_logger(__name__)

MIN_COVER = 0.6  # share of a DB block that must lie inside the classical block
MAX_GROWTH = 3.0  # a classical block may be at most this much larger than what it extends


class DbPrimaryDetector:
    name = "db_primary"

    def __init__(self, db: TextDetector, classical: TextDetector) -> None:
        self.db = db
        self.classical = classical

    def detect(self, rgb: RgbArray) -> list[TextBlock]:
        blocks = self.db.detect(rgb)
        try:
            extra = self.classical.detect(rgb)
        except DetectionError as exc:
            log.warning("classical detector failed (%s); using DB blocks only", exc)
            return blocks
        for cand in sorted(extra, key=lambda b: -b.bbox.area):
            covered = [b for b in blocks if b.bbox.overlap_ratio(cand.bbox) >= MIN_COVER]
            if not covered:
                continue
            base = sum(b.bbox.area for b in covered)
            if cand.bbox.area <= 1.1 * base or cand.bbox.area > MAX_GROWTH * base:
                continue  # nothing to add, or a texture blob swallowing real text
            for b in covered:
                blocks.remove(b)
            blocks.append(_merge(cand, covered))
        blocks.sort(key=lambda b: (b.bbox.y0, b.bbox.x0))
        return blocks
