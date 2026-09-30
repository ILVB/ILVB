"""Shared helpers turning ML detector boxes/masks into TextBlocks."""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

from manga_ar.detect.base import RgbArray, TextBlock
from manga_ar.detect.classical import ClassicalDetector
from manga_ar.detect.geometry import disk
from manga_ar.schemas import BBox

BoolArray = npt.NDArray[np.bool_]


def blocks_from_boxes(
    rgb: RgbArray, boxes: list[BBox], classical: ClassicalDetector, name: str
) -> list[TextBlock]:
    """Group line boxes into blocks and extract exact text pixels inside them.

    Boxes are grouped by dilation (≈ half a line thickness). Text pixels are the
    polarity-aware ink inside the boxes; the polarity is chosen per block by which ink
    covers the typical text fraction of the box area.
    """
    h, w = rgb.shape[:2]
    if not boxes:
        return []
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dark = classical.ink_mask(gray, "dark")
    light = classical.ink_mask(gray, "light")
    filled = np.zeros((h, w), np.uint8)
    for b in boxes:
        c = b.clip(w, h)
        filled[c.y0 : c.y1, c.x0 : c.x1] = 1
    thickness = float(np.median([min(b.width, b.height) for b in boxes]))
    joined = cv2.dilate(filled, disk(max(2, int(0.5 * thickness))))
    n, labels = cv2.connectedComponents(joined, connectivity=8)
    blocks: list[TextBlock] = []
    for i in range(1, n):
        region = (labels == i) & (filled > 0)
        dfrac = float(dark[region].mean()) if region.any() else 0.0
        lfrac = float(light[region].mean()) if region.any() else 0.0
        polarity = "dark" if abs(dfrac - 0.2) <= abs(lfrac - 0.2) else "light"
        glyphs = region & (dark if polarity == "dark" else light)
        block = classical.make_block(glyphs, name, polarity)
        if block is not None:
            blocks.append(block)
    return blocks


def blocks_from_mask(
    text_prob: npt.NDArray[np.float32],
    block_boxes: list[BBox],
    classical: ClassicalDetector,
    name: str,
    threshold: float = 0.35,
) -> list[TextBlock]:
    """Blocks from a pixel text-probability map restricted to detector block boxes."""
    h, w = text_prob.shape
    mask = text_prob >= threshold
    out: list[TextBlock] = []
    for b in block_boxes:
        c = b.clip(w, h)
        sub = np.zeros((h, w), bool)
        sub[c.y0 : c.y1, c.x0 : c.x1] = mask[c.y0 : c.y1, c.x0 : c.x1]
        block = classical.make_block(sub, name)
        if block is not None:
            out.append(block)
    return out
