"""Inpainter interface. Implementations fill ``mask`` pixels of an RGB crop."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

RgbArray = npt.NDArray[np.uint8]
BoolArray = npt.NDArray[np.bool_]


@runtime_checkable
class Inpainter(Protocol):
    name: str

    def available(self) -> bool: ...

    def inpaint(self, image: RgbArray, mask: BoolArray) -> RgbArray:
        """Return a same-shape image; only ``mask`` pixels may differ (callers enforce it)."""
        ...
