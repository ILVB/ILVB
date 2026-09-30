"""Vertical → horizontal reflow for engines that only read horizontal lines (E7).

A vertical CJK column is sliced into glyph cells along blank rows (touching glyphs are
split into roughly square cells), and the cells are concatenated left → right, keeping
top → bottom order. Several columns are joined right → left.
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

RgbArray = npt.NDArray[np.uint8]


def _ink_mask(img: RgbArray) -> npt.NDArray[np.bool_]:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # polarity: text is the minority colour relative to the border
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    bg = float(np.median(border))
    return np.asarray(np.abs(gray.astype(np.int16) - int(bg)) > 60, dtype=np.bool_)


def merge_stroke_runs(runs: list[tuple[int, int]], size: float) -> list[tuple[int, int]]:
    """Merge 1-D ink runs into glyph cells.

    Repeatedly take the shortest run below half a glyph and join it to the neighbour
    across the smaller gap, as long as the merged cell still fits one glyph (≤ 1.15 ×
    size). A stroke therefore attaches to the glyph it is visually closest to (う's dash
    to its curve, not to the glyph above), while two whole glyphs are never fused.
    """
    cells = list(runs)
    limit = 1.15 * size
    changed = True
    while changed and len(cells) > 1:
        changed = False
        order = sorted(range(len(cells)), key=lambda k: cells[k][1] - cells[k][0])
        for k in order:
            a, b = cells[k]
            if b - a >= 0.5 * size:
                break
            options = []
            if k > 0 and b - cells[k - 1][0] <= limit:
                options.append((a - cells[k - 1][1], k - 1))
            if k + 1 < len(cells) and cells[k + 1][1] - a <= limit:
                options.append((cells[k + 1][0] - b, k + 1))
            if not options:
                continue
            _, j = min(options)
            lo, hi = min(k, j), max(k, j)
            cells[lo : hi + 1] = [(cells[lo][0], cells[hi][1])]
            changed = True
            break
    return cells


def glyph_cells(column: RgbArray, size: float | None = None) -> list[tuple[int, int]]:
    """Row ranges of glyph cells in a vertical column image.

    Blank-row runs are merged into glyph cells by :func:`merge_stroke_runs`, so
    multi-stroke glyphs (こ, 三, う's dash) stay whole and neighbouring glyphs stay apart.
    Over-tall runs (touching glyphs) are split into square-ish pieces. ``size`` is the
    region's glyph size; it defaults to the column width.
    """
    ink = _ink_mask(column)
    h, w = ink.shape
    s = float(size) if size else float(w)
    rows = np.asarray(ink.any(axis=1), dtype=np.bool_).tolist()
    runs: list[tuple[int, int]] = []
    start = None
    for y in range(h):
        if rows[y] and start is None:
            start = y
        elif not rows[y] and start is not None:
            runs.append((start, y))
            start = None
    if start is not None:
        runs.append((start, h))
    cells = merge_stroke_runs(runs, s)
    out: list[tuple[int, int]] = []
    for a, b in cells:
        n = max(1, round((b - a) / max(1.0, 1.1 * s))) if (b - a) > 1.35 * s else 1
        step = (b - a) / n
        for k in range(n):
            out.append((a + round(k * step), a + round((k + 1) * step)))
    return out


def reflow_column(column: RgbArray, pad: int = 4, size: float | None = None) -> RgbArray:
    """Turn one vertical column into a horizontal strip of its glyph cells."""
    w = column.shape[1]
    cells = glyph_cells(column, size)
    if not cells:
        return column
    bg = np.median(np.concatenate([column[0], column[-1]]), axis=0).astype(np.uint8)
    size = int(max(w, max(b - a for a, b in cells)))
    pieces = []
    for a, b in cells:
        tile = np.empty((size, size, 3), np.uint8)
        tile[:] = bg
        glyph = column[a:b]
        oy = (size - glyph.shape[0]) // 2
        ox = (size - glyph.shape[1]) // 2
        tile[oy : oy + glyph.shape[0], ox : ox + glyph.shape[1]] = glyph
        pieces.append(tile)
        pieces.append(np.tile(bg, (size, max(1, size // 10), 1)))
    strip = np.concatenate(pieces[:-1], axis=1)
    border = np.empty((size + 2 * pad, strip.shape[1] + 2 * pad, 3), np.uint8)
    border[:] = bg
    border[pad : pad + size, pad : pad + strip.shape[1]] = strip
    return border


def reflow_columns(columns: list[RgbArray], size: float | None = None) -> RgbArray:
    """Reflow columns (already in reading order) into one horizontal line."""
    strips = [reflow_column(c, size=size) for c in columns]
    height = max(s.shape[0] for s in strips)
    padded = []
    for s in strips:
        if s.shape[0] != height:
            scale = height / s.shape[0]
            s = cv2.resize(
                s, (max(1, int(s.shape[1] * scale)), height), interpolation=cv2.INTER_LANCZOS4
            )
        padded.append(s)
    return np.concatenate(padded, axis=1)
