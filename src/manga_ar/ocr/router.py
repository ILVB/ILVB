"""OCR routing with fallbacks, upscaled retries, validation and language detection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from manga_ar.config import OcrConfig
from manga_ar.detect.geometry import disk
from manga_ar.errors import ModelUnavailableError, OcrError
from manga_ar.logging_setup import get_logger
from manga_ar.ocr.base import LineResult, OcrEngine, RgbArray
from manga_ar.ocr.langid import classify_text, is_passthrough, script_consistency
from manga_ar.ocr.postprocess import postprocess
from manga_ar.ocr.reflow import reflow_column
from manga_ar.ocr.suspicion import Verdict, assess
from manga_ar.schemas import BBox, Flag, OcrCandidate, OcrResult, Region, RegionType

log = get_logger(__name__)


@dataclass
class _Candidate:
    engine: str
    text: str
    confidence: float | None
    decorations: list[str]
    verdict: Verdict
    raw: str


class OcrRouter:
    """Selects engines per language, validates results and falls back on suspicion."""

    def __init__(self, engines: Mapping[str, OcrEngine], cfg: OcrConfig) -> None:
        self.engines = dict(engines)
        self.cfg = cfg
        self._broken: set[str] = set()  # engines that failed to load this run

    # ------------------------------------------------------------ engines
    def engines_for(self, lang: str) -> list[OcrEngine]:
        out = []
        for name in self.cfg.engines.get(lang, []):
            eng = self.engines.get(name)
            if eng is None or name in self._broken or lang not in eng.languages:
                continue
            if eng.available():
                out.append(eng)
        return out

    # --------------------------------------------------------------- crops
    def _crop(self, page: RgbArray, region: Region, upscale: float) -> tuple[RgbArray, BBox]:
        h, w = page.shape[:2]
        bb = region.bbox
        pad = max(3, int(self.cfg.padding * max(bb.width, bb.height)))
        box = bb.expand(pad).clip(w, h)
        crop = page[box.y0 : box.y1, box.x0 : box.x1].copy()
        # Paint out text pixels that are not in any recognised line (furigana, stray marks).
        if region.text_mask is not None and region.lines:
            keep = np.zeros((box.height, box.width), bool)
            for ln in region.lines:
                lb = ln.expand(2).intersection(box)
                if lb is not None:
                    keep[lb.y0 - box.y0 : lb.y1 - box.y0, lb.x0 - box.x0 : lb.x1 - box.x0] = True
            stray = region.text_mask.window(box) & ~keep
            if stray.any():
                stray = cv2.dilate(stray.astype(np.uint8), disk(1)) > 0
                crop[stray & ~keep] = region.fill_color or (255, 255, 255)
        if upscale > 1.0:
            crop = cv2.resize(crop, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_LANCZOS4)
        return crop, box

    def _line_images(self, page: RgbArray, region: Region, upscale: float) -> list[RgbArray]:
        h, w = page.shape[:2]
        images = []
        lines = region.lines or [region.bbox]
        size = float(np.median([b.width if region.vertical else b.height for b in lines]))
        for ln in lines:
            thickness = ln.width if region.vertical else ln.height
            pad = 3 + int(0.1 * thickness)
            box = ln.expand(pad).clip(w, h)
            img = page[box.y0 : box.y1, box.x0 : box.x1].copy()
            if region.vertical:
                img = reflow_column(img, size=size)
            scale = upscale
            if img.shape[0] < self.cfg.min_crop_px:
                scale = max(scale, min(self.cfg.max_upscale, self.cfg.min_crop_px / img.shape[0]))
            if scale > 1.0:
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
            images.append(img)
        return images

    def run_engine(
        self, engine: OcrEngine, page: RgbArray, region: Region, lang: str, upscale: float = 1.0
    ) -> LineResult:
        if engine.reads_blocks:
            crop, _ = self._crop(page, region, upscale)
            if min(crop.shape[:2]) < self.cfg.min_crop_px:
                f = min(self.cfg.max_upscale, self.cfg.min_crop_px / min(crop.shape[:2]))
                crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_LANCZOS4)
            return engine.recognize_block(crop, lang, region.vertical)
        results = [
            engine.recognize_line(img, lang) for img in self._line_images(page, region, upscale)
        ]
        sep = " " if lang in ("ko", "en") else ""  # line breaks are word breaks
        text = sep.join(r.text for r in results if r.text)
        confs = [r.confidence for r in results if r.confidence is not None]
        return LineResult(text, float(np.mean(confs)) if confs else None)

    # ----------------------------------------------------------- recognise
    def recognize(self, page: RgbArray, region: Region, lang: str) -> OcrResult:
        """OCR one region; sets OCR_SUSPECT / LOW_CONFIDENCE / OCR_FAILED flags."""
        engines = self.engines_for(lang)
        if not engines:
            region.flag(Flag.OCR_FAILED)
            return OcrResult(engine="none", text="", lang=lang, vertical=region.vertical)
        area = region.bbox.area
        glyph = (
            float(np.median([b.width if region.vertical else b.height for b in region.lines]))
            if region.lines
            else 0.0
        )
        candidates: list[_Candidate] = []
        attempts = [(e, 1.0) for e in engines]
        small = min(region.bbox.width, region.bbox.height) < 2 * self.cfg.min_crop_px
        if small and self.cfg.max_upscale > 1.0:
            attempts += [(e, min(2.0, self.cfg.max_upscale)) for e in engines]
        good: list[_Candidate] = []
        for engine, scale in attempts:
            try:
                raw = self.run_engine(engine, page, region, lang, scale)
            except ModelUnavailableError as exc:
                log.warning("OCR engine %s unavailable: %s", engine.name, exc)
                self._broken.add(engine.name)
                continue
            except OcrError as exc:
                log.warning("OCR engine %s failed on %s: %s", engine.name, region.id, exc)
                continue
            text, decorations = postprocess(raw.text, lang, self.cfg.preserve_decorations)
            verdict = assess(text, lang, raw.confidence, area, glyph, self.cfg.min_confidence)
            cand = _Candidate(engine.name, text, raw.confidence, decorations, verdict, raw.text)
            candidates.append(cand)
            if not verdict.suspicious:
                good.append(cand)
                if not self.cfg.verify_multi_engine or len(good) >= 2:
                    break
        if not candidates:
            region.flag(Flag.OCR_FAILED)
            return OcrResult(engine="none", text="", lang=lang, vertical=region.vertical)
        pool = good or candidates
        best = max(pool, key=lambda c: c.verdict.score)
        if not good:
            region.flag(Flag.OCR_SUSPECT)
            log.info("%s: OCR suspect (%s)", region.id, ", ".join(best.verdict.reasons))
        if best.confidence is not None and best.confidence < self.cfg.min_confidence:
            region.flag(Flag.LOW_CONFIDENCE)
        return OcrResult(
            engine=best.engine,
            text=best.text,
            confidence=best.confidence,
            lang=lang,
            vertical=region.vertical,
            raw_text=best.raw,
            decorations=best.decorations,
            alternatives=[
                OcrCandidate(c.engine, c.text, c.confidence, c.verdict.score)
                for c in candidates
                if c is not best
            ],
        )

    def recognize_all(self, page: RgbArray, regions: Sequence[Region], lang: str) -> None:
        for region in regions:
            if region.type == RegionType.SFX or region.override.skip:
                continue
            region_lang = region.override.source_lang or lang
            region.source_lang = region_lang
            region.ocr = self.recognize(page, region, region_lang)
            if self.cfg.pass_through_latin and region.ocr.text and is_passthrough(region.ocr.text):
                region.flag(Flag.PASS_THROUGH)

    # ------------------------------------------------------ language detection
    def detect_language(
        self, page: RgbArray, regions: Sequence[Region], max_samples: int = 4
    ) -> tuple[str | None, dict[str, float]]:
        """Pick the source language by running each language's first engine on a sample
        of the largest regions and scoring confidence × script agreement."""
        sample = sorted(
            (r for r in regions if r.type != RegionType.SFX),
            key=lambda r: -r.bbox.area,
        )[:max_samples]
        if not sample:
            return None, {}
        scores: dict[str, float] = {}
        vertical_share = sum(r.vertical for r in sample) / len(sample)
        for lang in ("ja", "ko", "zh"):
            engines = self.engines_for(lang)
            if not engines:
                continue
            total = 0.0
            for region in sample:
                try:
                    raw = self.run_engine(engines[0], page, region, lang)
                except (OcrError, ModelUnavailableError) as exc:
                    log.debug("language probe %s failed: %s", lang, exc)
                    continue
                text, _ = postprocess(raw.text, lang, False)
                if not text:
                    continue
                agree = 1.0 if classify_text(text) == lang else 0.5
                conf = raw.confidence if raw.confidence is not None else 0.6
                total += conf * script_consistency(text, lang) * agree
            scores[lang] = total / len(sample)
        if "ja" in scores:
            scores["ja"] += 0.15 * vertical_share  # vertical columns are a strong JA prior
        if not scores or max(scores.values()) <= 0.05:
            return None, scores
        return max(scores, key=lambda k: scores[k]), scores
