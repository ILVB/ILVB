"""manga-ocr adapter (Japanese; whole padded bubble crop, vertical or horizontal)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from PIL import Image

from manga_ar.errors import ModelUnavailableError, OcrError
from manga_ar.models.manager import ModelManager
from manga_ar.ocr.base import LineResult, RgbArray


class MangaOcrEngine:
    name = "manga_ocr"
    languages = frozenset({"ja"})
    reads_blocks = True

    def __init__(self, manager: ModelManager, device: str = "cpu") -> None:
        self.manager = manager
        self.device = device

    def available(self) -> bool:
        if importlib.util.find_spec("manga_ocr") is None:
            return False
        return self.manager.is_present("manga-ocr") or not self.manager.offline

    def _model(self) -> Any:
        def build(path: Path) -> Any:
            from manga_ocr import MangaOcr

            try:
                return MangaOcr(
                    pretrained_model_name_or_path=str(path), force_cpu=self.device == "cpu"
                )
            except (OSError, RuntimeError, ValueError) as exc:
                raise ModelUnavailableError(f"manga-ocr could not load: {exc}") from exc

        return self.manager.load("manga-ocr", build)

    def recognize_block(self, block: RgbArray, lang: str, vertical: bool) -> LineResult:
        model = self._model()
        try:
            text = model(Image.fromarray(block))
        except (RuntimeError, ValueError) as exc:
            raise OcrError(f"manga-ocr failed: {exc}") from exc
        return LineResult(str(text), None)

    def recognize_line(self, line: RgbArray, lang: str) -> LineResult:
        return self.recognize_block(line, lang, vertical=False)
