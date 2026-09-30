"""OCR engine interface.

Engines expose line-level recognition (``recognize_line``: one horizontal text line →
text + confidence). Engines that read a whole multi-line/vertical block natively
(manga-ocr) also set ``reads_blocks`` and implement ``recognize_block``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

RgbArray = npt.NDArray[np.uint8]


@dataclass
class LineResult:
    text: str
    confidence: float | None


@runtime_checkable
class OcrEngine(Protocol):
    name: str
    languages: frozenset[str]
    reads_blocks: bool

    def available(self) -> bool:
        """Cheap check: package importable and model obtainable (no heavy loading)."""
        ...

    def recognize_line(self, line: RgbArray, lang: str) -> LineResult:
        """Recognise one horizontal line image (RGB). Raises OcrError on failure."""
        ...

    def recognize_block(self, block: RgbArray, lang: str, vertical: bool) -> LineResult:
        """Recognise a whole block (only if ``reads_blocks``)."""
        ...
