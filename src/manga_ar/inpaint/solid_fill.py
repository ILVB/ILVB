"""Solid fill with the median background colour (uniform bubbles)."""

from __future__ import annotations

import cv2
import numpy as np

from manga_ar.detect.geometry import disk
from manga_ar.inpaint.base import BoolArray, RgbArray


def ring_sample(
    image: RgbArray, mask: BoolArray, zone: BoolArray | None, inner: int = 2, outer: int = 8
) -> RgbArray:
    """Background pixels in a ring around ``mask`` (inside ``zone`` when given)."""
    m = mask.astype(np.uint8)
    ring = (cv2.dilate(m, disk(outer)) > 0) & ~(cv2.dilate(m, disk(inner)) > 0)
    if zone is not None:
        ring &= zone
    return np.asarray(image[ring], dtype=np.uint8)


class SolidFill:
    name = "solid"

    def __init__(self, zone: BoolArray | None = None, feather: int = 1) -> None:
        self.zone = zone
        self.feather = feather

    def available(self) -> bool:
        return True

    def inpaint(self, image: RgbArray, mask: BoolArray) -> RgbArray:
        out = image.copy()
        if not mask.any():
            return out
        sample = ring_sample(image, mask, self.zone)
        if sample.size == 0:
            sample = image[~mask] if (~mask).any() else image.reshape(-1, 3)
        color = np.median(sample.reshape(-1, 3), axis=0)
        out[mask] = np.rint(color).astype(np.uint8)
        if self.feather > 0:
            # 1-px feather on the mask's inner edge: blend the fill with the pixels that
            # were there, but only where those already look like background (never
            # re-introduce anti-aliased glyph remnants).
            edge = mask & ~(cv2.erode(mask.astype(np.uint8), disk(self.feather)) > 0)
            near = np.abs(image.astype(np.int16) - color.astype(np.int16)).max(axis=2) <= 24
            soft = edge & near
            blended = 0.5 * image[soft].astype(np.float32) + 0.5 * color
            out[soft] = np.rint(blended).astype(np.uint8)
        return out
