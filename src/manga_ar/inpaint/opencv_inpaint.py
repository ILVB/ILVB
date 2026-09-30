"""OpenCV Telea / Navier-Stokes inpainting (mandatory baseline, no model)."""

from __future__ import annotations

import cv2
import numpy as np

from manga_ar.inpaint.base import BoolArray, RgbArray


class OpenCvInpainter:
    """``method``: "telea" or "ns". The radius adapts to the mask's thickness."""

    def __init__(self, method: str = "telea", radius: int | None = None) -> None:
        if method not in {"telea", "ns"}:
            raise ValueError(f"unknown OpenCV inpaint method {method!r}")
        self.name = method
        self.radius = radius

    def available(self) -> bool:
        return True

    def inpaint(self, image: RgbArray, mask: BoolArray) -> RgbArray:
        if not mask.any():
            return image.copy()
        m8 = mask.astype(np.uint8) * 255
        radius = self.radius
        if radius is None:
            dt = cv2.distanceTransform(m8, cv2.DIST_L2, 3)
            radius = int(np.clip(round(float(dt.max()) + 2), 3, 15))
        flag = cv2.INPAINT_TELEA if self.name == "telea" else cv2.INPAINT_NS
        bgr = np.ascontiguousarray(image[..., ::-1])
        out = cv2.inpaint(bgr, m8, radius, flag)
        return np.ascontiguousarray(np.asarray(out, dtype=np.uint8)[..., ::-1])
