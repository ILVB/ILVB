"""1.0.3: DB-primary detector merge rules (fake detectors, no models)."""

from __future__ import annotations

import numpy as np

from manga_ar.detect.base import TextBlock
from manga_ar.detect.db_primary import DbPrimaryDetector
from manga_ar.errors import DetectionError
from manga_ar.schemas import BBox, CropMask


def _block(x0: int, y0: int, x1: int, y1: int, name: str) -> TextBlock:
    box = BBox(x0, y0, x1, y1)
    return TextBlock(box, [box], text_mask=CropMask(box, np.ones((y1 - y0, x1 - x0), bool)),
                     detector=name, glyph_size=20)  # fmt: skip


class Fixed:
    def __init__(self, blocks: list[TextBlock] | Exception) -> None:
        self.blocks = blocks

    def detect(self, rgb: np.ndarray) -> list[TextBlock]:
        if isinstance(self.blocks, Exception):
            raise self.blocks
        return [TextBlock(b.bbox, list(b.lines), text_mask=b.text_mask, detector=b.detector,
                          glyph_size=b.glyph_size) for b in self.blocks]  # fmt: skip


RGB = np.zeros((400, 400, 3), np.uint8)


def test_classical_block_extends_a_partial_db_block() -> None:
    db = Fixed([_block(269, 100, 297, 250, "rapid")])  # right column only
    classical = Fixed([_block(233, 100, 297, 250, "classical")])  # both columns
    (block,) = DbPrimaryDetector(db, classical).detect(RGB)  # type: ignore[arg-type]
    assert block.bbox == BBox(233, 100, 297, 250)
    assert block.text_mask is not None and block.text_mask.area == 64 * 150


def test_classical_never_adds_standalone_blocks() -> None:
    db = Fixed([_block(10, 10, 60, 40, "rapid")])
    classical = Fixed([_block(10, 10, 60, 40, "classical"), _block(200, 200, 260, 260, "c")])
    blocks = DbPrimaryDetector(db, classical).detect(RGB)  # type: ignore[arg-type]
    assert [b.bbox for b in blocks] == [BBox(10, 10, 60, 40)]  # same size: nothing to add


def test_texture_blob_swallowing_text_is_ignored() -> None:
    db = Fixed([_block(100, 100, 130, 120, "rapid")])
    blob = Fixed([_block(50, 50, 350, 350, "classical")])  # 150x larger
    blocks = DbPrimaryDetector(db, blob).detect(RGB)  # type: ignore[arg-type]
    assert [b.bbox for b in blocks] == [BBox(100, 100, 130, 120)]


def test_classical_failure_keeps_db_blocks() -> None:
    db = Fixed([_block(10, 10, 60, 40, "rapid")])
    blocks = DbPrimaryDetector(db, Fixed(DetectionError("x"))).detect(RGB)  # type: ignore[arg-type]
    assert len(blocks) == 1
