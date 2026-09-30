"""1.0.4: DB line geometry for blocks and texture suppression on OCR crops."""

from __future__ import annotations

import numpy as np

from manga_ar.config import load_config
from manga_ar.detect.adapters import blocks_from_boxes
from manga_ar.detect.classical import ClassicalDetector
from manga_ar.ocr.preprocess import suppress_texture
from manga_ar.schemas import BBox

CFG = load_config(environ={})


def _two_line_page(dots: bool) -> np.ndarray:
    img = np.full((160, 300, 3), 255, np.uint8)
    if dots:  # light 2x2 screentone dots everywhere (real tone dots are 2-4 px)
        for dy in (0, 1):
            for dx in (0, 1):
                img[dy::6, dx::6] = 170
    for y in (40, 90):
        for x in range(40, 250, 26):
            img[y : y + 20, x : x + 18] = 0  # glyph-like blobs on two text lines
    return img


def test_db_boxes_become_the_block_lines() -> None:
    img = _two_line_page(dots=True)
    boxes = [BBox(36, 36, 254, 64), BBox(36, 86, 254, 114)]
    classical = ClassicalDetector(CFG.detect)
    (block,) = blocks_from_boxes(img, boxes, classical, "rapid", geometry_from_boxes=True)
    assert block.vertical is False and block.lines == boxes
    (legacy,) = blocks_from_boxes(img, boxes, classical, "rapid")  # legacy: glyph segmentation
    assert legacy.lines != boxes


def test_texture_is_removed_but_strokes_and_flat_crops_stay() -> None:
    textured = _two_line_page(dots=True)[30:70]
    clean = suppress_texture(textured)
    assert clean is not textured
    assert (clean[10:30, 40:58] < 60).mean() > 0.9  # glyph strokes kept (first glyph)
    background = np.ones(clean.shape[:2], bool)
    background[3:37, 26:278] = False  # glyphs span x 40..264 on rows 10..30
    assert clean[background].min() == 255  # dots away from glyphs are gone
    flat = _two_line_page(dots=False)[30:70]
    assert suppress_texture(flat) is flat  # no texture: untouched
    tiny = np.zeros((4, 4, 3), np.uint8)
    assert suppress_texture(tiny) is tiny
