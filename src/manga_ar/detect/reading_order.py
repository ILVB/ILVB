"""Reading order (affects translation context only, never placement).

Panels are found by recursive cuts along blank gutter bands of the page image; regions
are ordered by panel, then inside a panel by a recursive XY-cut over region boxes that
prefers horizontal cuts (top → bottom) and orders columns right → left for manga.
"""

from __future__ import annotations

import itertools

import cv2
import numpy as np
import numpy.typing as npt

from manga_ar.schemas import BBox, Region

U8 = npt.NDArray[np.uint8]
MODES = ("manga_rtl", "comic_ltr", "webtoon_ttb")


def default_mode(lang: str | None) -> str:
    return {"ja": "manga_rtl", "zh": "comic_ltr", "ko": "webtoon_ttb"}.get(lang or "", "manga_rtl")


def _bands(profile: npt.NDArray[np.float64], min_len: int) -> list[tuple[int, int]]:
    """Maximal runs of ``True`` (blank) positions at least ``min_len`` long."""
    out: list[tuple[int, int]] = []
    start = None
    for i, blank in enumerate(profile):
        if blank and start is None:
            start = i
        elif not blank and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    if start is not None and len(profile) - start >= min_len:
        out.append((start, len(profile)))
    return out


def find_panels(gray: U8, mode: str, min_gutter: int = 8, depth: int = 0) -> list[BBox]:
    """Recursive gutter cuts on a light page; returns panels in reading order."""
    h, w = gray.shape
    return _cut(gray, BBox(0, 0, w, h), mode, min_gutter, depth)


def _cut(gray: U8, box: BBox, mode: str, min_gutter: int, depth: int) -> list[BBox]:
    if depth > 6 or box.width < 4 * min_gutter or box.height < 4 * min_gutter:
        return [box]
    sub = gray[box.y0 : box.y1, box.x0 : box.x1]
    light = sub >= 235
    rows_blank = light.mean(axis=1) >= 0.985
    cols_blank = light.mean(axis=0) >= 0.985
    for axis in ("rows", "cols"):
        blank = rows_blank if axis == "rows" else cols_blank
        bands = [b for b in _bands(blank, min_gutter) if b[0] > 0 and b[1] < len(blank)]
        if not bands:
            continue
        cuts = [0] + [(a + b) // 2 for a, b in bands] + [len(blank)]
        parts: list[BBox] = []
        for a, b in itertools.pairwise(cuts):
            if b - a < 2 * min_gutter:
                continue
            if axis == "rows":
                parts.append(BBox(box.x0, box.y0 + a, box.x1, box.y0 + b))
            else:
                parts.append(BBox(box.x0 + a, box.y0, box.x0 + b, box.y1))
        if len(parts) < 2:
            continue
        if axis == "cols" and mode == "manga_rtl":
            parts.reverse()
        out: list[BBox] = []
        for part in parts:
            out.extend(_cut(gray, part, mode, min_gutter, depth + 1))
        return out
    return [box]


def _xy_order(boxes: list[tuple[int, BBox]], mode: str) -> list[int]:
    if len(boxes) <= 1:
        return [i for i, _ in boxes]
    for axis in ("y", "x"):
        spans = sorted(((b.y0, b.y1) if axis == "y" else (b.x0, b.x1), i) for i, b in boxes)
        groups: list[list[int]] = []
        current: list[int] = []
        reach = -1
        for (lo, hi), i in spans:
            if current and lo >= reach:
                groups.append(current)
                current = []
            current.append(i)
            reach = max(reach, hi)
        groups.append(current)
        if len(groups) > 1:
            if axis == "x" and mode == "manga_rtl":
                groups.reverse()
            order: list[int] = []
            lookup = dict(boxes)
            for g in groups:
                order.extend(_xy_order([(i, lookup[i]) for i in g], mode))
            return order
    # fully overlapping: fall back to centre sort
    key = (
        (lambda ib: (ib[1].center[1], -ib[1].center[0]))
        if mode == "manga_rtl"
        else (lambda ib: (ib[1].center[1], ib[1].center[0]))
    )
    return [i for i, _ in sorted(boxes, key=key)]


def assign_reading_order(regions: list[Region], rgb: U8, mode: str) -> list[Region]:
    """Set ``reading_order`` on every region and return them sorted by it.

    webtoon_ttb reads panels (full-width bands in a strip, a grid on printed manhwa)
    top→bottom / left→right, and regions inside a panel purely top→bottom.
    """
    if not regions:
        return regions
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    panels = find_panels(gray, "comic_ltr" if mode == "webtoon_ttb" else mode)
    buckets: dict[int, list[tuple[int, BBox]]] = {}
    for idx, r in enumerate(regions):
        cx, cy = r.bbox.center
        pid = next(
            (k for k, p in enumerate(panels) if p.x0 <= cx < p.x1 and p.y0 <= cy < p.y1),
            len(panels),
        )
        buckets.setdefault(pid, []).append((idx, r.bbox))
    order: list[int] = []
    for pid in sorted(buckets):
        if mode == "webtoon_ttb":
            ranked = sorted(buckets[pid], key=lambda ib: (ib[1].center[1], ib[1].center[0]))
            order.extend(i for i, _ in ranked)
        else:
            order.extend(_xy_order(buckets[pid], mode))
    ordered = [regions[i] for i in order]
    for k, r in enumerate(ordered):
        r.reading_order = k
    return ordered
