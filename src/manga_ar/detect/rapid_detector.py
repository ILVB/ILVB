"""PP-OCR DB text-line detector (bundled with rapidocr_onnxruntime, Apache-2.0)."""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np

from manga_ar.config import DetectConfig
from manga_ar.detect.adapters import blocks_from_boxes
from manga_ar.detect.base import RgbArray, TextBlock
from manga_ar.detect.classical import ClassicalDetector
from manga_ar.errors import DetectionError, ModelUnavailableError
from manga_ar.schemas import BBox


class RapidDbDetector:
    name = "rapid"

    def __init__(self, cfg: DetectConfig, classical: ClassicalDetector) -> None:
        if importlib.util.find_spec("rapidocr_onnxruntime") is None:
            raise ModelUnavailableError("rapidocr_onnxruntime is not installed (extra: rapid)")
        from rapidocr_onnxruntime import RapidOCR

        self.cfg = cfg
        self.classical = classical
        self._engine: Any = RapidOCR()

    def detect(self, rgb: RgbArray) -> list[TextBlock]:
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        try:
            quads, _ = self._engine(bgr, use_det=True, use_cls=False, use_rec=False)
        except (RuntimeError, ValueError) as exc:
            raise DetectionError(f"RapidOCR detection failed: {exc}") from exc
        boxes = []
        for quad in quads or []:
            q = np.asarray(quad, dtype=np.float64)
            boxes.append(
                BBox(
                    int(q[:, 0].min()),
                    int(q[:, 1].min()),
                    int(np.ceil(q[:, 0].max())),
                    int(np.ceil(q[:, 1].max())),
                )
            )
        return blocks_from_boxes(rgb, boxes, self.classical, self.name)
