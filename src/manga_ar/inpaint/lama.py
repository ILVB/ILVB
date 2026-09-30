"""LaMa (big-lama TorchScript) inpainting with context crops and CPU/OOM fallback."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from manga_ar.errors import InpaintError, ModelUnavailableError
from manga_ar.inpaint.base import BoolArray, RgbArray
from manga_ar.logging_setup import get_logger
from manga_ar.models.device import run_with_cpu_fallback
from manga_ar.models.manager import ModelManager

log = get_logger(__name__)


class LamaInpainter:
    """Runs on a crop the caller already padded with context; resizes to fit
    ``max_side`` (multiples of 8) and back. Output is the model result for the whole crop."""

    name = "lama"

    def __init__(self, manager: ModelManager, device: str = "cpu", max_side: int = 1024) -> None:
        self.manager = manager
        self.device = device
        self.max_side = max_side

    def available(self) -> bool:
        if importlib.util.find_spec("torch") is None:
            return False
        return self.manager.is_present("lama") or not self.manager.offline

    def _model(self) -> Any:
        def build(path: Path) -> Any:
            import torch

            try:
                return torch.jit.load(str(path), map_location="cpu").eval()
            except (RuntimeError, OSError, ValueError) as exc:
                raise ModelUnavailableError(f"LaMa model could not load: {exc}") from exc

        return self.manager.load("lama", build)

    def inpaint(self, image: RgbArray, mask: BoolArray) -> RgbArray:
        if not mask.any():
            return image.copy()
        model = self._model()
        h, w = image.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        nh = max(8, int(np.ceil(h * scale / 8.0)) * 8)
        nw = max(8, int(np.ceil(w * scale / 8.0)) * 8)
        img = cv2.resize(
            image, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        )
        msk = cv2.resize(mask.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST)
        msk = cv2.dilate(msk, np.ones((3, 3), np.uint8))  # cover resize rounding

        def run(device: str) -> RgbArray:
            import torch

            m = model.to(device)
            ti = torch.from_numpy(img).permute(2, 0, 1)[None].float().div(255.0).to(device)
            tm = torch.from_numpy(msk)[None, None].float().to(device)
            with torch.inference_mode():
                out = m(ti, tm)
            res = out[0].permute(1, 2, 0).detach().cpu().numpy()
            return np.asarray(np.clip(res * 255.0, 0, 255).astype(np.uint8))

        try:
            result, used = run_with_cpu_fallback(run, self.device, "lama")
        except (RuntimeError, MemoryError) as exc:
            raise InpaintError(f"LaMa failed: {exc}") from exc
        if used != self.device:
            self.device = used
        return np.asarray(cv2.resize(result, (w, h), interpolation=cv2.INTER_CUBIC), dtype=np.uint8)
