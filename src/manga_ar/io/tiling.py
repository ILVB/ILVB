"""Vertical tiling for long strips (webtoons) with gutter-aware cuts and seam merging.

Detection runs per tile; every later stage works on page coordinates using crops of the
full page, so stitching is lossless by construction. Detections duplicated across a seam
are merged so each text block is processed exactly once, preferring the copy from a tile
that contains it completely.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

import cv2
import numpy as np
import numpy.typing as npt

from manga_ar.schemas import BBox

T = TypeVar("T")


@dataclass(frozen=True)
class Tile:
    y0: int
    y1: int

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def needs_tiling(height: int, width: int, aspect_trigger: float, height_trigger: int) -> bool:
    return height > aspect_trigger * width or height > height_trigger


def row_energy(gray: npt.NDArray[np.uint8]) -> npt.NDArray[np.float64]:
    """Per-row edge energy: mean |d/dx| + |d/dy|. Low values mark gutters/blank rows."""
    g = gray.astype(np.float32)
    dx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))
    dy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    energy = (dx + dy).mean(axis=1).astype(np.float64)
    # Smooth over a few rows so a single blank row inside text is not mistaken for a gutter.
    kernel = np.ones(9, dtype=np.float64) / 9.0
    return np.convolve(energy, kernel, mode="same")


def plan_tiles(
    gray: npt.NDArray[np.uint8], tile_height: int, overlap: int, search: int | None = None
) -> list[Tile]:
    """Cut ``gray`` (H×W) into tiles ≤ ``tile_height`` with overlap ≥ ``overlap``.

    Each cut row is the lowest-energy row in the last ``search`` rows of the window, so cuts
    prefer gutters. The next tile starts ``overlap`` rows above the cut.
    """
    height = int(gray.shape[0])
    if height <= tile_height:
        return [Tile(0, height)]
    search = search if search is not None else max(overlap, tile_height // 4)
    search = min(search, tile_height - 2 * overlap - 1)
    energy = row_energy(gray)
    tiles: list[Tile] = []
    y0 = 0
    while True:
        limit = y0 + tile_height
        if limit >= height:
            tiles.append(Tile(y0, height))
            break
        lo = max(y0 + 2 * overlap + 1, limit - search)
        window = energy[lo:limit]
        cut = lo + int(np.argmin(window)) if window.size else limit
        tiles.append(Tile(y0, cut))
        y0 = cut - overlap
    return tiles


def is_complete_in_tile(box: BBox, tile: Tile, page_height: int, margin: int = 2) -> bool:
    """True when ``box`` does not touch an interior cut of ``tile``."""
    top_ok = tile.y0 == 0 or box.y0 > tile.y0 + margin
    bottom_ok = tile.y1 >= page_height or box.y1 < tile.y1 - margin
    return top_ok and bottom_ok


def merge_tiled(
    items: Sequence[tuple[Tile, T]],
    box_of: Callable[[T], BBox],
    page_height: int,
    iou_threshold: float = 0.3,
    containment: float = 0.6,
) -> list[tuple[T, bool]]:
    """Deduplicate detections from overlapping tiles.

    Returns ``(item, complete)`` pairs. Items overlapping by IoU ≥ ``iou_threshold`` or
    with one covering ≥ ``containment`` of the other form one group; the representative is
    the largest item that is complete in its tile, else the largest item overall
    (``complete=False`` then signals a block cut by every seam).
    """
    n = len(items)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    boxes = [box_of(item) for _, item in items]
    for i in range(n):
        for j in range(i + 1, n):
            if items[i][0] == items[j][0]:
                continue  # same tile: the detector already de-duplicated these
            a, b = boxes[i], boxes[j]
            if (
                a.iou(b) >= iou_threshold
                or a.overlap_ratio(b) >= containment
                or b.overlap_ratio(a) >= containment
            ):
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out: list[tuple[T, bool]] = []
    for members in groups.values():
        complete = [i for i in members if is_complete_in_tile(boxes[i], items[i][0], page_height)]
        pool = complete or members
        best = max(pool, key=lambda i: (boxes[i].area, -boxes[i].y0))
        out.append((items[best][1], bool(complete)))
    out.sort(key=lambda pair: (box_of(pair[0]).y0, box_of(pair[0]).x0))
    return out
