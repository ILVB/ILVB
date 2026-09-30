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


ASPECT = 1.5  # a line box this much wider than tall (or taller than wide) fixes orientation


def box_orientation(boxes: list[BBox]) -> bool | None:
    """Orientation from detector line boxes: False (horizontal), True (vertical), or None
    when the boxes are too square to tell (e.g. a lone "!")."""
    if not boxes:
        return None
    ratios = sorted(b.width / max(1, b.height) for b in boxes)
    median = ratios[len(ratios) // 2]
    if median >= ASPECT:
        return False
    if median <= 1 / ASPECT:
        return True
    return None


def blocks_from_boxes(
    rgb: RgbArray, boxes: list[BBox], classical: ClassicalDetector, name: str,
    geometry_from_boxes: bool = False,
) -> list[TextBlock]:  # fmt: skip
    """Group line boxes into blocks and extract exact text pixels inside them.

    Boxes are grouped by dilation (≈ half a line thickness). Text pixels are the
    polarity-aware ink inside the boxes; the polarity is chosen per block by which ink
    covers the typical text fraction of the box area. With ``geometry_from_boxes`` (v2), a
    block whose line boxes are clearly wide or tall takes that orientation, and the boxes
    themselves as its lines, instead of re-segmenting a glyph mask that may hold screentone
    dots or hatching.
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
        vertical, lines = None, None
        if geometry_from_boxes:
            members = [b.clip(w, h) for b in boxes if _label_at(labels, b.clip(w, h)) == i]
            vertical = box_orientation(members)
            lines = members if vertical is not None else None
        block = classical.make_block(glyphs, name, polarity, vertical=vertical, lines=lines)
        if block is not None:
            blocks.append(block)
    return blocks


def _label_at(labels: np.ndarray, box: BBox) -> int:
    cx, cy = box.center
    return int(labels[min(int(cy), labels.shape[0] - 1), min(int(cx), labels.shape[1] - 1)])


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
