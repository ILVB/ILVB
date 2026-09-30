"""EasyOCR CRAFT text detector (weights from GitHub releases)."""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np

from manga_ar.config import DetectConfig
from manga_ar.detect.adapters import blocks_from_boxes
from manga_ar.detect.base import RgbArray, TextBlock
from manga_ar.detect.classical import ClassicalDetector
from manga_ar.errors import DetectionError, ModelUnavailableError
from manga_ar.models.manager import ModelManager
from manga_ar.schemas import BBox


class CraftDetector:
    name = "craft"

    def __init__(
        self,
        cfg: DetectConfig,
        manager: ModelManager,
        classical: ClassicalDetector,
        offline: bool = False,
    ) -> None:
        if importlib.util.find_spec("easyocr") is None:
            raise ModelUnavailableError("easyocr is not installed (extra: ocr)")
        import easyocr

        model_dir = manager.ensure("easyocr")
        if offline and not (model_dir / "craft_mlt_25k.pth").is_file():
            raise ModelUnavailableError("CRAFT weights missing and offline mode is on")
        self.cfg = cfg
        self.classical = classical
        try:
            self._reader: Any = easyocr.Reader(
                ["en"],
                gpu=False,
                model_storage_directory=str(model_dir),
                download_enabled=not offline,
                recognizer=False,
                verbose=False,
            )
        except (OSError, RuntimeError) as exc:
            raise ModelUnavailableError(f"CRAFT could not load: {exc}") from exc

    def detect(self, rgb: RgbArray) -> list[TextBlock]:
        try:
            horizontal, free = self._reader.detect(rgb)
        except (RuntimeError, ValueError) as exc:
            raise DetectionError(f"CRAFT detection failed: {exc}") from exc
        boxes = [BBox(int(b[0]), int(b[2]), int(b[1]), int(b[3])) for b in horizontal[0]]
        for poly in free[0]:
            p = np.asarray(poly, dtype=np.float64)
            boxes.append(
                BBox(
                    int(p[:, 0].min()),
                    int(p[:, 1].min()),
                    int(np.ceil(p[:, 0].max())),
                    int(np.ceil(p[:, 1].max())),
                )
            )
        boxes = [b for b in boxes if b.width > 0 and b.height > 0]
        return blocks_from_boxes(rgb, boxes, self.classical, self.name)
