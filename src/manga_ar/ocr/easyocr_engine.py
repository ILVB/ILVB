"""EasyOCR adapter (ja / ko / zh). Weights come from GitHub releases on first use."""

from __future__ import annotations

import importlib.util
import threading
from typing import Any

import cv2
import numpy as np

from manga_ar.errors import ModelUnavailableError, OcrError
from manga_ar.logging_setup import get_logger
from manga_ar.models.manager import ModelManager
from manga_ar.ocr.base import LineResult, RgbArray

log = get_logger(__name__)

# EasyOCR language groups must be combined legally (never ja+ko+ch together).
LANG_GROUPS: dict[str, list[str]] = {"ja": ["ja", "en"], "ko": ["ko", "en"], "zh": ["ch_sim", "en"]}
_MODEL_FILES = {"ja": "japanese_g2.pth", "ko": "korean_g2.pth", "zh": "zh_sim_g2.pth"}


class EasyOcrEngine:
    name = "easyocr"
    languages = frozenset({"ja", "ko", "zh"})
    reads_blocks = False

    def __init__(self, manager: ModelManager, device: str = "cpu", offline: bool = False) -> None:
        self.manager = manager
        self.device = device
        self.offline = offline
        self._readers: dict[str, Any] = {}
        self._lock = threading.Lock()

    def available(self) -> bool:
        if importlib.util.find_spec("easyocr") is None:
            return False
        if not self.offline:
            return True
        return any((self.manager.path("easyocr") / f).is_file() for f in _MODEL_FILES.values())

    def _reader(self, lang: str) -> Any:
        with self._lock:
            if lang in self._readers:
                return self._readers[lang]
            model_dir = self.manager.ensure("easyocr")
            if self.offline and not (model_dir / _MODEL_FILES[lang]).is_file():
                raise ModelUnavailableError(
                    f"EasyOCR model for {lang!r} not downloaded and offline mode is on; run "
                    "`manga-arabic models download easyocr` while online"
                )
            import easyocr

            try:
                reader = easyocr.Reader(
                    LANG_GROUPS[lang],
                    gpu=self.device == "cuda",
                    model_storage_directory=str(model_dir),
                    download_enabled=not self.offline,
                    detector=False,
                    verbose=False,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                raise ModelUnavailableError(f"EasyOCR ({lang}) could not load: {exc}") from exc
            self._readers[lang] = reader
            return reader

    def recognize_line(self, line: RgbArray, lang: str) -> LineResult:
        reader = self._reader(lang)
        gray = cv2.cvtColor(line, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        try:
            res = reader.recognize(
                gray, horizontal_list=[[0, w, 0, h]], free_list=[], detail=1, paragraph=False
            )
        except (RuntimeError, ValueError, IndexError) as exc:
            raise OcrError(f"EasyOCR recognition failed: {exc}") from exc
        texts = [str(r[1]) for r in res]
        confs = [float(r[2]) for r in res if len(r) > 2]
        return LineResult("".join(texts), float(np.mean(confs)) if confs else None)

    def recognize_block(self, block: RgbArray, lang: str, vertical: bool) -> LineResult:
        return self.recognize_line(block, lang)


def download_easyocr_models(
    manager: ModelManager, languages: tuple[str, ...] = ("ja", "ko", "zh")
) -> None:
    """Fetch CRAFT + recognisers for ``languages`` into the MangaAR cache."""
    import easyocr

    model_dir = manager.ensure("easyocr")
    for lang in languages:
        easyocr.Reader(
            LANG_GROUPS[lang],
            gpu=False,
            model_storage_directory=str(model_dir),
            download_enabled=True,
            verbose=False,
        )
