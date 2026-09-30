"""Binary mask run-length encoding for benchmark files (independent of manga_ar).

A mask is stored as its tight bounding box plus row-major runs inside it, starting
with a run of zeros: ``{"bbox": [x0, y0, x1, y1], "runs": [...]}`` (x1/y1 exclusive).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

BoolArray = npt.NDArray[np.bool_]


def encode(mask: BoolArray) -> dict[str, Any] | None:
    """Encode a full-size mask; ``None`` when it is empty."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    flat = mask[y0:y1, x0:x1].astype(np.uint8).ravel()
    edges = np.flatnonzero(np.diff(flat)) + 1
    bounds = np.concatenate(([0], edges, [flat.size]))
    runs = np.diff(bounds).tolist()
    if flat[0] == 1:
        runs = [0, *runs]
    return {"bbox": [x0, y0, x1, y1], "runs": [int(r) for r in runs]}


def decode(obj: dict[str, Any] | None, shape: tuple[int, int]) -> BoolArray:
    out = np.zeros(shape, dtype=bool)
    if obj is None:
        return out
    x0, y0, x1, y1 = (int(v) for v in obj["bbox"])
    flat = np.zeros((y1 - y0) * (x1 - x0), dtype=bool)
    pos, value = 0, False
    for run in obj["runs"]:
        flat[pos : pos + run] = value
        pos += run
        value = not value
    if pos != flat.size:
        raise ValueError(f"RLE covers {pos} pixels, bbox has {flat.size}")
    out[y0:y1, x0:x1] = flat.reshape(y1 - y0, x1 - x0)
    return out
