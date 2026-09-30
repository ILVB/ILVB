"""Mask geometry helpers shared by detection, inpainting and typesetting."""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

from manga_ar.schemas import BBox

BoolArray = npt.NDArray[np.bool_]


def largest_inscribed_rect(mask: BoolArray) -> BBox | None:
    """Largest axis-aligned rectangle of ``True`` pixels (maximal-rectangle algorithm).

    Row-by-row histogram + monotonic stack: O(H·W). Returns ``None`` for an empty mask.
    """
    h, w = mask.shape
    if h == 0 or w == 0:
        return None
    heights = np.zeros(w + 1, dtype=np.int64)  # trailing sentinel column of height 0
    best = (0, 0, 0, 0, 0)  # area, x0, y0, x1, y1
    for y in range(h):
        row = mask[y]
        heights[:w] = np.where(row, heights[:w] + 1, 0)
        stack: list[int] = []
        for x in range(w + 1):
            while stack and heights[stack[-1]] >= heights[x]:
                top = stack.pop()
                height = int(heights[top])
                left = stack[-1] + 1 if stack else 0
                area = height * (x - left)
                if area > best[0]:
                    best = (area, left, y - height + 1, x, y + 1)
            stack.append(x)
    if best[0] == 0:
        return None
    _, x0, y0, x1, y1 = best
    return BBox(x0, y0, x1, y1)


def fill_holes(mask: BoolArray) -> BoolArray:
    """Fill interior holes (e.g. glyph counters) of a binary mask."""
    m = mask.astype(np.uint8)
    h, w = m.shape
    padded = np.zeros((h + 2, w + 2), np.uint8)
    padded[1:-1, 1:-1] = m
    flood = padded.copy()
    ff_mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 1)
    holes = flood[1:-1, 1:-1] == 0
    return np.asarray(mask | holes, dtype=np.bool_)


def solidity(mask: BoolArray) -> float:
    """Area / convex-hull area of the largest component (1.0 = convex)."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return 0.0
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    hull = cv2.contourArea(cv2.convexHull(c))
    return float(area / hull) if hull > 0 else 0.0


def largest_component(mask: BoolArray) -> BoolArray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask.copy()
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.asarray(labels == idx, dtype=np.bool_)


def mask_polygon(mask: BoolArray, epsilon_frac: float = 0.01) -> list[tuple[int, int]]:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    c = max(contours, key=cv2.contourArea)
    eps = epsilon_frac * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps, True)
    return [(int(p[0][0]), int(p[0][1])) for p in approx]


def ellipse_mask(shape: tuple[int, int], box: BBox) -> BoolArray:
    """Filled ellipse inscribed in ``box`` (clipped to ``shape``)."""
    out = np.zeros(shape, np.uint8)
    center = (round(box.center[0]), round(box.center[1]))
    axes = (max(1, box.width // 2), max(1, box.height // 2))
    cv2.ellipse(out, center, axes, 0, 0, 360, 1, -1)
    return out.astype(bool)


def disk(radius: int) -> npt.NDArray[np.uint8]:
    size = 2 * max(0, radius) + 1
    return np.asarray(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)), dtype=np.uint8)


def erode(mask: BoolArray, radius: int) -> BoolArray:
    """Disk erosion treating everything outside the array as background.

    OpenCV's default border counts outside pixels as foreground, so a mask touching its
    crop edges (every tight bubble mask does, at its extremes) would keep full-width
    spikes reaching the outline there."""
    if radius <= 0:
        return mask.copy()
    out = cv2.erode(
        mask.astype(np.uint8), disk(radius), borderType=cv2.BORDER_CONSTANT, borderValue=0
    )
    return np.asarray(out > 0, dtype=np.bool_)
