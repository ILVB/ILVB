"""Typesetting metrics on ground-truth safe regions (G-TYPE-1, -2, -4)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

Bool = npt.NDArray[np.bool_]
HYPHENS = ("-", "‐", "‑", "­")


def ink_outside(ink: Bool, safe: Bool) -> int:
    """G-TYPE-1: rendered ink pixels outside the ground-truth safe region."""
    return int((ink & ~safe).sum())


def available_width(safe: Bool, y0: float, y1: float) -> float:
    """Longest horizontal run of the safe region, minimised over the line's rows."""
    h = safe.shape[0]
    a, b = max(0, int(np.floor(y0))), min(h, int(np.ceil(y1)))
    if b <= a:
        return 0.0
    widths = []
    for row in safe[a:b]:
        if not row.any():
            return 0.0
        padded = np.concatenate(([False], row, [False]))
        edges = np.flatnonzero(padded[1:] != padded[:-1]).reshape(-1, 2)
        widths.append(int((edges[:, 1] - edges[:, 0]).max()))
    return float(min(widths))


def raggedness(line_boxes: Sequence[Sequence[float]], safe: Bool) -> float | None:
    """Mean squared relative slack of all lines but the last (lower is better)."""
    if len(line_boxes) < 2:
        return None
    slacks = []
    for x0, y0, x1, y1 in line_boxes[:-1]:
        avail = available_width(safe, y0, y1)
        if avail <= 0:
            slacks.append(1.0)
            continue
        slacks.append(max(0.0, avail - (x1 - x0)) / avail)
    return float(np.mean(np.square(slacks)))


def is_orphan(lines: Sequence[str]) -> bool:
    """Widow/orphan: a multi-line block whose last line is a single word."""
    return len(lines) >= 2 and len(lines[-1].split()) == 1


def arabic_hyphen_breaks(lines: Sequence[str]) -> int:
    """Hyphenated line breaks in Arabic text (never valid, G-TYPE-4)."""
    return sum(1 for ln in lines[:-1] if ln.rstrip().endswith(HYPHENS))


def above_floor(size_px: float, page_height: int, floor_frac: float = 0.011) -> bool:
    """G-TYPE-2: size at or above the legibility floor (default 1.1 % of page height)."""
    return size_px + 1e-9 >= floor_frac * page_height
