"""DB-primary detector (v2): PP-OCR DB decides which blocks exist; classical refines them.

PP-OCR DB (bundled with RapidOCR) finds text over screentone and artwork where connected
components fail, and rejects most texture false positives. Classical blocks are precise
on clean backgrounds but absorb texture on screentone and art. So (ADR-0004):
- a classical block never enters on its own;
- on a clean background, a classical block that agrees with a DB block (IoU >= 0.7)
  replaces it (its line/column geometry reads better overall: taking it only for vertical
  blocks regressed screentone CER on dev), and a larger classical block may *extend* the
  DB blocks it covers (e.g. the short last column of vertical text);
- on a textured background the DB block is kept as is.
"""

from __future__ import annotations

from manga_ar.detect.base import RgbArray, TextBlock, TextDetector
from manga_ar.detect.hybrid import _merge, _textured_background
from manga_ar.errors import DetectionError
from manga_ar.logging_setup import get_logger

log = get_logger(__name__)

AGREE_IOU = 0.7  # classical and DB found the same block
MIN_COVER = 0.6  # share of a DB block that must lie inside an extending classical block
MAX_GROWTH = 3.0  # an extending block may be at most this much larger than what it covers


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
            if _textured_background(rgb, cand):
                continue
            same = [b for b in blocks if b.vertical == cand.vertical]  # never flip orientation
            agree = [b for b in same if b.bbox.iou(cand.bbox) >= AGREE_IOU]
            if agree:
                for b in agree:
                    blocks.remove(b)
                blocks.append(cand)
                continue
            covered = [b for b in same if b.bbox.overlap_ratio(cand.bbox) >= MIN_COVER]
            if not covered:
                continue
            base = sum(b.bbox.area for b in covered)
            if cand.bbox.area <= 1.1 * base or cand.bbox.area > MAX_GROWTH * base:
                continue
            for b in covered:
                blocks.remove(b)
            blocks.append(_merge(cand, covered))
        blocks.sort(key=lambda b: (b.bbox.y0, b.bbox.x0))
        return blocks
