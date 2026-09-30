"""comic-text-detector (ONNX) adapter: text blocks + pixel text mask.

The weights are GPL-3.0 (zyddnys/manga-image-translator release), so this detector is
opt-in (``detect.detector: ctd``). This module is original code: it only runs the ONNX
graph and post-processes its outputs.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from manga_ar.config import DetectConfig
from manga_ar.detect.adapters import blocks_from_mask
from manga_ar.detect.base import RgbArray, TextBlock
from manga_ar.detect.classical import ClassicalDetector
from manga_ar.errors import DetectionError, ModelUnavailableError
from manga_ar.models.manager import ModelManager
from manga_ar.schemas import BBox

INPUT = 1024


class ComicTextDetector:
    name = "ctd"

    def __init__(
        self, cfg: DetectConfig, manager: ModelManager, classical: ClassicalDetector
    ) -> None:
        if importlib.util.find_spec("onnxruntime") is None:
            raise ModelUnavailableError("onnxruntime is not installed (extra: ctd)")
        self.cfg = cfg
        self.classical = classical

        def build(path: Path) -> Any:
            import onnxruntime as ort

            return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

        self._session: Any = manager.load("ctd", build)

    def detect(self, rgb: RgbArray) -> list[TextBlock]:
        h, w = rgb.shape[:2]
        scale = INPUT / max(h, w)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        canvas = np.zeros((INPUT, INPUT, 3), np.uint8)
        canvas[:nh, :nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        blob = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        try:
            blk, seg, _lines = self._session.run(None, {self._session.get_inputs()[0].name: blob})
        except (RuntimeError, ValueError) as exc:
            raise DetectionError(f"comic-text-detector failed: {exc}") from exc
        preds = blk[0]
        preds = preds[preds[:, 4] > 0.4]
        rects = [
            [float(x - bw / 2), float(y - bh / 2), float(bw), float(bh)]
            for x, y, bw, bh in preds[:, :4]
        ]
        keep = (
            cv2.dnn.NMSBoxes(rects, preds[:, 4].astype(float).tolist(), 0.4, 0.45) if rects else []
        )
        boxes = []
        for i in np.asarray(keep).flatten():
            x, y, bw, bh = rects[int(i)]
            boxes.append(
                BBox(
                    int(x / scale),
                    int(y / scale),
                    int((x + bw) / scale) + 1,
                    int((y + bh) / scale) + 1,
                ).clip(w, h)
            )
        prob = cv2.resize(seg[0, 0][:nh, :nw], (w, h), interpolation=cv2.INTER_LINEAR)
        return blocks_from_mask(prob.astype(np.float32), boxes, self.classical, self.name)
