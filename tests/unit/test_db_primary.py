"""1.0.3: DB-primary detector merge rules (fake detectors, no models)."""

from __future__ import annotations

import numpy as np

from manga_ar.detect.base import TextBlock
from manga_ar.detect.db_primary import DbPrimaryDetector
from manga_ar.errors import DetectionError
from manga_ar.schemas import BBox, CropMask


def _block(x0: int, y0: int, x1: int, y1: int, name: str, vertical: bool = False) -> TextBlock:
    box = BBox(x0, y0, x1, y1)
    return TextBlock(box, [box], text_mask=CropMask(box, np.ones((y1 - y0, x1 - x0), bool)),
                     detector=name, glyph_size=20, vertical=vertical)  # fmt: skip


class Fixed:
    def __init__(self, blocks: list[TextBlock] | Exception) -> None:
        self.blocks = blocks

    def detect(self, rgb: np.ndarray) -> list[TextBlock]:
        if isinstance(self.blocks, Exception):
            raise self.blocks
        return [TextBlock(b.bbox, list(b.lines), text_mask=b.text_mask, detector=b.detector,
                          glyph_size=b.glyph_size, vertical=b.vertical)
                for b in self.blocks]  # fmt: skip


RGB = np.full((400, 400, 3), 255, np.uint8)  # clean background


def _textured() -> np.ndarray:
    img = RGB.copy()
    img[::3, :] = 0  # hatching everywhere
    return img


def test_classical_block_extends_a_partial_db_block() -> None:
    db = Fixed([_block(269, 100, 297, 250, "rapid")])  # right column only
    classical = Fixed([_block(233, 100, 297, 250, "classical")])  # both columns
    (block,) = DbPrimaryDetector(db, classical).detect(RGB)  # type: ignore[arg-type]
    assert block.bbox == BBox(233, 100, 297, 250)
    assert block.text_mask is not None and block.text_mask.area == 64 * 150


def test_classical_never_adds_standalone_blocks() -> None:
    db = Fixed([_block(10, 10, 60, 40, "rapid")])
    classical = Fixed([_block(200, 200, 260, 260, "classical")])
    blocks = DbPrimaryDetector(db, classical).detect(RGB)  # type: ignore[arg-type]
    assert [(b.bbox, b.detector) for b in blocks] == [(BBox(10, 10, 60, 40), "rapid")]


def test_agreeing_classical_block_supplies_the_geometry_on_clean_backgrounds() -> None:
    for vertical in (True, False):
        db = Fixed([_block(10, 10, 40, 60, "rapid", vertical=vertical)])
        classical = Fixed([_block(10, 11, 41, 60, "classical", vertical=vertical)])
        (clean,) = DbPrimaryDetector(db, classical).detect(RGB)  # type: ignore[arg-type]
        assert clean.detector == "classical"
        flipped = Fixed([_block(10, 11, 41, 60, "classical", vertical=not vertical)])
        (kept,) = DbPrimaryDetector(db, flipped).detect(RGB)  # type: ignore[arg-type]
        assert kept.detector == "rapid"  # a classical block never flips the orientation
    db = Fixed([_block(10, 10, 40, 60, "rapid")])
    classical = Fixed([_block(10, 11, 41, 60, "classical")])
    (textured,) = DbPrimaryDetector(db, classical).detect(_textured())  # type: ignore[arg-type]
    assert textured.detector == "rapid"  # texture-polluted classical geometry is not used
    (textured,) = DbPrimaryDetector(db, classical).detect(_textured())  # type: ignore[arg-type]
    assert textured.detector == "rapid"  # texture-polluted classical geometry is not used


def test_no_extension_on_textured_background() -> None:
    db = Fixed([_block(269, 100, 297, 250, "rapid")])
    classical = Fixed([_block(233, 100, 297, 250, "classical")])
    (block,) = DbPrimaryDetector(db, classical).detect(_textured())  # type: ignore[arg-type]
    assert block.bbox == BBox(269, 100, 297, 250)


def test_texture_blob_swallowing_text_is_ignored() -> None:
    db = Fixed([_block(100, 100, 130, 120, "rapid")])
    blob = Fixed([_block(50, 50, 350, 350, "classical")])  # 150x larger
    blocks = DbPrimaryDetector(db, blob).detect(RGB)  # type: ignore[arg-type]
    assert [b.bbox for b in blocks] == [BBox(100, 100, 130, 120)]


def test_classical_failure_keeps_db_blocks() -> None:
    db = Fixed([_block(10, 10, 60, 40, "rapid")])
    blocks = DbPrimaryDetector(db, Fixed(DetectionError("x"))).detect(RGB)  # type: ignore[arg-type]
    assert len(blocks) == 1


def test_auto_resolves_by_profile() -> None:
    from manga_ar.config import load_config
    from manga_ar.detect.factory import resolve_detector

    assert resolve_detector(load_config(environ={})) == "classical"  # legacy default
    v2 = load_config(overrides={"engine.profile": "v2"}, environ={})
    assert resolve_detector(v2) == "db_primary"
    pinned = load_config(overrides={"engine.profile": "v2", "detect.detector": "ctd"}, environ={})
    assert resolve_detector(pinned) == "ctd"  # an explicit choice always wins


def test_orientation_from_line_boxes() -> None:
    from manga_ar.detect.adapters import box_orientation

    assert box_orientation([BBox(0, 0, 90, 20), BBox(0, 30, 80, 50)]) is False  # wide lines
    assert box_orientation([BBox(0, 0, 20, 90)]) is True  # a tall column
    assert box_orientation([BBox(0, 0, 22, 20)]) is None  # "!" alone: undecided
    assert box_orientation([]) is None
