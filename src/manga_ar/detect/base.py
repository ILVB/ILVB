"""Stage interfaces and data exchanged by text detectors and bubble segmenters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from manga_ar.schemas import BBox, CropMask, Region

RgbArray = npt.NDArray[np.uint8]


@dataclass
class TextBlock:
    """A group of text lines believed to belong together (one speech unit)."""

    bbox: BBox
    lines: list[BBox] = field(default_factory=list)
    vertical: bool = False
    score: float = 1.0
    text_mask: CropMask | None = None
    detector: str = "unknown"
    glyph_size: float = 0.0  # median glyph extent (px)
    is_sfx: bool = False
    polarity: str = "dark"  # "dark" text on light, or "light" text on dark
    furigana: list[BBox] = field(default_factory=list)

    def translate(self, dx: int, dy: int) -> TextBlock:
        return TextBlock(
            bbox=self.bbox.translate(dx, dy),
            lines=[b.translate(dx, dy) for b in self.lines],
            vertical=self.vertical,
            score=self.score,
            text_mask=None if self.text_mask is None else self.text_mask.translate(dx, dy),
            detector=self.detector,
            glyph_size=self.glyph_size,
            is_sfx=self.is_sfx,
            polarity=self.polarity,
            furigana=[b.translate(dx, dy) for b in self.furigana],
        )


@runtime_checkable
class TextDetector(Protocol):
    """Finds text blocks on an RGB page (or tile)."""

    name: str

    def detect(self, rgb: RgbArray) -> list[TextBlock]: ...


@runtime_checkable
class BubbleSegmenter(Protocol):
    """Turns text blocks into typed regions with bubble masks and safe areas."""

    def segment(self, rgb: RgbArray, blocks: list[TextBlock], page_id: str) -> list[Region]: ...
