"""RapidOCR adapter: PP-OCR (Chinese) ONNX models bundled in ``rapidocr_onnxruntime``."""

from __future__ import annotations

import importlib.util
import threading
from typing import Any

import numpy as np

from manga_ar.errors import ModelUnavailableError, OcrError
from manga_ar.ocr.base import LineResult, RgbArray


class RapidOcrEngine:
    name = "rapid"
    languages = frozenset({"zh"})  # bundled model is Chinese; kana/Hangul are not covered
    reads_blocks = False

    def __init__(self, num_threads: int = 0) -> None:
        self.num_threads = num_threads
        self._engine: Any = None
        self._lock = threading.Lock()

    def available(self) -> bool:
        return importlib.util.find_spec("rapidocr_onnxruntime") is not None

    def _get(self) -> Any:
        with self._lock:
            if self._engine is None:
                try:
                    from rapidocr_onnxruntime import RapidOCR

                    self._engine = RapidOCR()
                except (ImportError, OSError, RuntimeError) as exc:
                    raise ModelUnavailableError(f"RapidOCR unavailable: {exc}") from exc
            return self._engine

    def recognize_line(self, line: RgbArray, lang: str) -> LineResult:
        engine = self._get()
        bgr = np.ascontiguousarray(line[..., ::-1])  # RapidOCR expects OpenCV BGR arrays
        try:
            res, _ = engine(bgr, use_det=False, use_cls=False, use_rec=True)
        except (RuntimeError, ValueError, IndexError) as exc:
            raise OcrError(f"RapidOCR recognition failed: {exc}") from exc
        if not res:
            return LineResult("", 0.0)
        texts = [str(r[0]) for r in res]
        confs = [float(r[1]) for r in res]
        return LineResult("".join(texts), float(np.mean(confs)))

    def recognize_block(self, block: RgbArray, lang: str, vertical: bool) -> LineResult:
        return self.recognize_line(block, lang)
