"""PaddleOCR adapter (2.x ``ocr()`` and 3.x ``predict()`` APIs).

Not verifiable on the build host (all Paddle model hosts blocked, DECISIONS W-002); the
output normaliser is unit-tested against both documented result layouts.
"""

from __future__ import annotations

import importlib.util
import threading
from collections.abc import Mapping
from typing import Any

import numpy as np

from manga_ar.errors import ModelUnavailableError, OcrError
from manga_ar.ocr.base import LineResult, RgbArray

PADDLE_LANG = {"ja": "japan", "ko": "korean", "zh": "ch"}


def normalize_paddle_output(res: Any) -> list[tuple[str, float]]:
    """Flatten PaddleOCR results into ``(text, score)`` pairs.

    3.x: a list of result objects/dicts exposing ``rec_texts`` and ``rec_scores``.
    2.x: ``[[ [box, (text, score)], ... ]]`` (one inner list per image, may be ``None``).
    """
    out: list[tuple[str, float]] = []
    if res is None:
        return out
    items = res if isinstance(res, list) else [res]
    for item in items:
        if item is None:
            continue
        data = item
        if not isinstance(item, Mapping) and hasattr(item, "json"):
            data = item.json
            data = data.get("res", data) if isinstance(data, Mapping) else item
        if isinstance(data, Mapping) and "rec_texts" in data:
            scores = list(data.get("rec_scores", [])) or [1.0] * len(data["rec_texts"])
            out.extend((str(t), float(s)) for t, s in zip(data["rec_texts"], scores, strict=False))
            continue
        if isinstance(data, list):
            for entry in data:
                if (
                    isinstance(entry, (list, tuple))
                    and len(entry) == 2
                    and isinstance(entry[1], (list, tuple))
                    and len(entry[1]) == 2
                ):
                    out.append((str(entry[1][0]), float(entry[1][1])))
                elif isinstance(entry, list):
                    out.extend(normalize_paddle_output([entry]))
    return out


class PaddleOcrEngine:
    name = "paddle"
    languages = frozenset({"ja", "ko", "zh"})
    reads_blocks = False

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._engines: dict[str, Any] = {}
        self._lock = threading.Lock()

    def available(self) -> bool:
        return importlib.util.find_spec("paddleocr") is not None

    def _engine(self, lang: str) -> Any:
        with self._lock:
            if lang in self._engines:
                return self._engines[lang]
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise ModelUnavailableError("paddleocr is not installed") from exc
            try:
                engine = PaddleOCR(
                    lang=PADDLE_LANG[lang],
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                )
            except TypeError:
                engine = PaddleOCR(lang=PADDLE_LANG[lang], use_angle_cls=False, show_log=False)
            except (RuntimeError, OSError, ValueError) as exc:
                raise ModelUnavailableError(f"PaddleOCR ({lang}) unavailable: {exc}") from exc
            self._engines[lang] = engine
            return engine

    def recognize_line(self, line: RgbArray, lang: str) -> LineResult:
        engine = self._engine(lang)
        bgr = np.ascontiguousarray(line[..., ::-1])
        try:
            res = engine.predict(bgr) if hasattr(engine, "predict") else engine.ocr(bgr, cls=False)
        except (RuntimeError, ValueError, IndexError) as exc:
            raise OcrError(f"PaddleOCR failed: {exc}") from exc
        pairs = normalize_paddle_output(res)
        if not pairs:
            return LineResult("", 0.0)
        return LineResult("".join(t for t, _ in pairs), float(np.mean([s for _, s in pairs])))

    def recognize_block(self, block: RgbArray, lang: str, vertical: bool) -> LineResult:
        return self.recognize_line(block, lang)
