"""In-test fakes for every pipeline stage (fast, deterministic, no models).

Pages are drawn by :func:`page`: bubble outlines with solid black "text" rectangles.
A rectangle's width encodes its source text (see ``CODES``), so the fake OCR can read
it back and the fake translator can map it to Arabic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np

from manga_ar.config import AppConfig
from manga_ar.detect.base import TextBlock
from manga_ar.errors import InpaintError, OcrError
from manga_ar.pipeline import Stages
from manga_ar.schemas import (
    BBox,
    CropMask,
    Flag,
    OcrResult,
    Region,
    RegionType,
    TranslationResult,
)
from manga_ar.typeset.fonts import FontRegistry
from manga_ar.typeset.layout import Typesetter

CODES = {40: "你好", 60: "再见", 80: "谢谢", 50: "没有翻译"}
ARABIC = {"你好": "مرحبا", "再见": "إلى اللقاء", "谢谢": "شكرا جزيلا"}
_REGISTRY = FontRegistry()


def page(
    widths: Sequence[int] = (40, 60), h: int = 300, w: int = 420, gray: bool = False
) -> np.ndarray:
    """White page; one bubble per width, each holding a black code rectangle."""
    img = np.full((h, w, 3), 255, np.uint8)
    for k, width in enumerate(widths):
        cx = 90 + k * 160
        cv2.ellipse(img, (cx, 150), (75, 95), 0, 0, 360, (0, 0, 0), 2)
        img[135:165, cx - width // 2 : cx - width // 2 + width] = 0
    if not gray:
        img[5:15, 5:15] = (200, 30, 30)  # a coloured mark: the page is not grayscale
    return img


class FakeDetector:
    name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, rgb: np.ndarray) -> list[TextBlock]:
        self.calls += 1
        dark = (rgb.max(axis=2) < 60).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
        blocks = []
        for i in range(1, n):
            x, y, bw, bh, area = (int(v) for v in stats[i])
            if area < 0.8 * bw * bh or area < 100:  # outlines are sparse; code blocks solid
                continue
            box = BBox(x, y, x + bw, y + bh)
            mask = CropMask(box, labels[y : y + bh, x : x + bw] == i)
            blocks.append(TextBlock(box, [box], text_mask=mask, glyph_size=bh, is_sfx=bh > 100))
        return blocks


class FakeSegmenter:
    def segment(self, rgb: np.ndarray, blocks: list[TextBlock], page_id: str) -> list[Region]:
        h, w = rgb.shape[:2]
        regions = []
        for k, b in enumerate(blocks):
            bubble = BBox(b.bbox.x0 - 30, b.bbox.y0 - 40, b.bbox.x1 + 30, b.bbox.y1 + 40).clip(w, h)
            regions.append(
                Region(
                    id=f"{page_id}-r{k}",
                    type=RegionType.SFX if b.is_sfx else RegionType.BUBBLE,
                    bbox=b.bbox,
                    lines=list(b.lines),
                    bubble_mask=CropMask(bubble, np.ones((bubble.height, bubble.width), bool)),
                    text_mask=b.text_mask,
                    fill_color=(255, 255, 255),
                )
            )
        return regions


@dataclass
class FakeOcr:
    fail_widths: set[int] = field(default_factory=set)
    lang_scores: dict[str, float] = field(default_factory=lambda: {"zh": 0.9, "ja": 0.2})
    probes: int = 0

    def recognize(self, page: np.ndarray, region: Region, lang: str) -> OcrResult:
        width = region.bbox.width
        if width in self.fail_widths:
            raise OcrError("fake OCR crash")
        return OcrResult(engine="fake", text=CODES.get(width, "???"), lang=lang)

    def detect_language(
        self, page: np.ndarray, regions: Sequence[Region], max_samples: int = 4
    ) -> tuple[str | None, dict[str, float]]:
        self.probes += 1
        return max(self.lang_scores, key=lambda k: self.lang_scores[k]), dict(self.lang_scores)


@dataclass
class FakeInpainter:
    fail_widths: set[int] = field(default_factory=set)

    def eligible(self, region: Region) -> bool:
        if region.type == RegionType.SFX or region.override.skip:
            return False
        return Flag.PASS_THROUGH not in region.flags

    def inpaint_region(self, page: np.ndarray, region: Region) -> str:
        assert region.text_mask is not None
        box = region.text_mask.bbox.expand(2).clip(page.shape[1], page.shape[0])
        region.inpaint_mask = CropMask(box, np.ones((box.height, box.width), bool))
        if region.bbox.width in self.fail_widths:
            raise InpaintError("fake inpaint failure")
        page[box.y0 : box.y1, box.x0 : box.x1] = 255
        return "solid"


@dataclass
class FakeTranslator:
    table: dict[str, str] = field(default_factory=lambda: dict(ARABIC))
    offline: bool = True
    calls: int = 0
    sources: list[str] = field(default_factory=list)

    def translate_regions(self, regions: Sequence[Region], src: str) -> None:
        self.calls += 1
        self.sources.append(src)
        for r in regions:
            if r.type == RegionType.SFX or r.override.skip or Flag.PASS_THROUGH in r.flags:
                continue
            text = r.ocr.text if r.ocr is not None else ""
            if text in self.table:
                r.translation = TranslationResult("fake", self.table[text])
                r.flags.discard(Flag.UNTRANSLATED)
            else:
                r.flag(Flag.UNTRANSLATED)

    def translate_text(self, text: str, src: str) -> TranslationResult | None:
        return TranslationResult("fake", self.table[text]) if text in self.table else None

    def has_offline_provider(self, src: str) -> bool:
        return self.offline


def fake_stages(cfg: AppConfig, **kwargs: object) -> Stages:
    return Stages(
        detector=kwargs.get("detector") or FakeDetector(),  # type: ignore[arg-type]
        segmenter=FakeSegmenter(),
        ocr=kwargs.get("ocr") or FakeOcr(),  # type: ignore[arg-type]
        inpainter=kwargs.get("inpainter") or FakeInpainter(),  # type: ignore[arg-type]
        translator=kwargs.get("translator") or FakeTranslator(),  # type: ignore[arg-type]
        typesetter=Typesetter(cfg.typeset, _REGISTRY),
    )
