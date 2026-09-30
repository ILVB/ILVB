"""Adapter for the frozen v0.1.0 API (runs with `.baseline/src` first on sys.path).

Only APIs that exist at tag v0.1.0 are used here, so this module keeps working against the
baseline forever. Stage functions return version-independent `benchmarks.schema` objects.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from benchmarks import rle
from benchmarks.adapters.meter import measure
from benchmarks.schema import GtPage, PageResult, PredRegion, StageStats, TypesetReport

RgbArray = npt.NDArray[np.uint8]
TYPE_TO_PROMPT = {"bubble": "dialogue", "narration": "narration", "free_text": "sign", "sfx": "sfx"}
PROMPT_TO_V010 = {
    "dialogue": "bubble",
    "thought": "bubble",
    "narration": "narration",
    "sign": "free_text",
    "credit": "free_text",
    "sfx": "sfx",
}
SOURCE_LANGS = {"ja", "ko", "zh"}  # v0.1.0 users pick "auto" for anything else (e.g. English)


class Engine:
    """Stage entry points over one configured v0.1.0 pipeline (models load lazily)."""

    def __init__(self, overrides: dict[str, Any], tm_pairs: dict[str, str] | None = None) -> None:
        import manga_ar
        from manga_ar.config import load_config
        from manga_ar.pipeline import Pipeline

        self.origin = str(Path(manga_ar.__file__).resolve().parent)
        # Only the model cache location comes from the environment; every other setting is
        # explicit so runs are reproducible (MANGAAR_OFFLINE, MANGAAR_DEVICE etc. ignored).
        environ = {k: os.environ[k] for k in ("MANGAAR_CACHE_DIR",) if k in os.environ}
        self.cfg = load_config(overrides=overrides, environ=environ)
        self.pipeline = Pipeline(self.cfg, tm_pairs=tm_pairs)

    # ------------------------------------------------------------------ helpers
    def _seed(self) -> None:
        random.seed(self.cfg.runtime.seed)
        np.random.seed(self.cfg.runtime.seed % 2**32)

    def _detect(
        self, rgb: RgbArray, page_id: str, lang: str, stats: dict[str, StageStats]
    ) -> tuple[Any, list[Any], str]:
        from manga_ar.detect.reading_order import assign_reading_order, default_mode
        from manga_ar.detect.tiled import detect_page
        from manga_ar.schemas import PageDocument, RegionType

        stages = self.pipeline.stages
        doc = PageDocument(source=page_id, width=rgb.shape[1], height=rgb.shape[0])
        with measure(stats, "detect"):
            blocks = detect_page(rgb, stages.detector, self.cfg.tiling)
            regions = stages.segmenter.segment(rgb, blocks, page_id)
            active = [r for r in regions if r.type != RegionType.SFX]
            sfx = [r for r in regions if r.type == RegionType.SFX]
            if lang not in SOURCE_LANGS:  # what `--source auto` does in v0.1.0 (one page)
                found = stages.ocr.detect_language(rgb, active)[0] if active else None
                lang = found or "ja"
            mode = self.cfg.input.reading_order
            active = (
                assign_reading_order(active, rgb, default_mode(lang) if mode == "auto" else mode)
                if active
                else []
            )
            for k, r in enumerate(sfx):
                r.reading_order = len(active) + k
        doc.regions = [*active, *sfx]
        doc.lang = lang
        return doc, active, lang

    @staticmethod
    def _pred(regions: list[Any], height: int, width: int) -> list[PredRegion]:
        out = []
        for r in regions:
            b = r.bbox
            mask = r.text_mask.to_full(height, width) if r.text_mask is not None else None
            erase = r.inpaint_mask.to_full(height, width) if r.inpaint_mask is not None else None
            out.append(
                PredRegion(
                    region_id=r.id,
                    polygon=[(b.x0, b.y0), (b.x1, b.y0), (b.x1, b.y1), (b.x0, b.y1)],
                    text=r.ocr.text if r.ocr is not None else "",
                    confidence=r.ocr.confidence if r.ocr is not None else None,
                    type=TYPE_TO_PROMPT.get(r.type.value, r.type.value),
                    lang=r.source_lang,
                    vertical=bool(r.vertical),
                    reading_order=int(r.reading_order),
                    text_mask=rle.encode(mask) if mask is not None else None,
                    erase_mask=rle.encode(erase) if erase is not None else None,
                    flags=sorted(f.value for f in r.flags),
                )
            )
        return out

    # ------------------------------------------------------------------ stages
    def detect_ocr(self, rgb: RgbArray, page_id: str, lang: str) -> PageResult:
        self._seed()
        stats: dict[str, StageStats] = {}
        doc, active, lang = self._detect(rgb, page_id, lang, stats)
        with measure(stats, "ocr"):
            self.pipeline._ocr(rgb, active, lang, doc)
        return PageResult(
            page_id=page_id,
            version="",
            mode="detect_ocr",
            regions=self._pred(doc.regions, *rgb.shape[:2]),
            stages=stats,
            code_origin=self.origin,
        )

    def erase(self, rgb: RgbArray, page_id: str, lang: str) -> tuple[PageResult, RgbArray]:
        self._seed()
        stats: dict[str, StageStats] = {}
        doc, active, lang = self._detect(rgb, page_id, lang, stats)
        with measure(stats, "ocr"):
            self.pipeline._ocr(rgb, active, lang, doc)
        with measure(stats, "inpaint"):
            clean, _failures = self.pipeline._inpaint(rgb, active, doc)
        result = PageResult(
            page_id=page_id,
            version="",
            mode="erase",
            regions=self._pred(doc.regions, *rgb.shape[:2]),
            stages=stats,
            code_origin=self.origin,
        )
        return result, clean

    def typeset_gt(
        self, original: RgbArray, clean: RgbArray, gt: GtPage, texts: dict[str, str]
    ) -> tuple[PageResult, RgbArray]:
        """Typeset fixed Arabic ``texts`` into ground-truth regions on the GT background."""
        from manga_ar.schemas import (
            BBox,
            CropMask,
            PageDocument,
            Region,
            RegionType,
            TranslationResult,
        )
        from manga_ar.typeset.page import typeset_page
        from manga_ar.typeset.render import render_layer

        self._seed()
        h, w = clean.shape[:2]
        doc = PageDocument(source=gt.page_id, width=w, height=h)
        for g in gt.regions:
            text_mask = rle.decode(g.text_mask, (h, w))
            ys, xs = np.nonzero(text_mask)
            box = BBox(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            bubble = None
            fill = None
            if g.bubble_mask is not None:
                full = rle.decode(g.bubble_mask, (h, w))
                bubble = CropMask.from_full(full)
                med = np.median(clean[full], axis=0)
                fill = (int(med[0]), int(med[1]), int(med[2]))
            region = Region(
                id=g.region_id,
                type=RegionType(PROMPT_TO_V010[g.type]),
                bbox=box,
                lines=[box],
                bubble_mask=bubble,
                text_mask=CropMask.from_full(text_mask),
                inpaint_mask=CropMask.from_full(text_mask),
                reading_order=g.reading_order,
                fill_color=fill,
            )
            if g.region_id in texts:
                region.translation = TranslationResult(provider="harness", text=texts[g.region_id])
            doc.regions.append(region)
        stats: dict[str, StageStats] = {}
        with measure(stats, "typeset"):
            render = typeset_page(
                doc,
                original,
                clean,
                self.pipeline.stages.typesetter,
                erase_untranslated=True,
            )
        reports = []
        for region in doc.regions:
            placement = render.placements.get(region.id)
            if placement is None:
                reports.append(TypesetReport(region_id=region.id, typeset=False))
                continue
            ink = render_layer((h, w), [placement])[..., 3] > 0
            boxes = []
            for line in placement.lines:
                top, bottom = placement.engine.ink_extent(
                    line.visual, placement.size, placement.outline_px
                )
                x0 = line.x - placement.outline_px
                boxes.append([x0, line.y + top, x0 + line.width, line.y + bottom])
            reports.append(
                TypesetReport(
                    region_id=region.id,
                    typeset=True,
                    font=region.layout.font if region.layout else "",
                    size_px=float(placement.size),
                    lines=[ln.logical for ln in placement.lines],
                    line_boxes=boxes,
                    ink_mask=rle.encode(ink),
                    overflow=bool(placement.overflow),
                    needs_review=bool(placement.overflow),
                    flags=sorted(f.value for f in region.flags),
                )
            )
        result = PageResult(
            page_id=gt.page_id,
            version="",
            mode="typeset_gt",
            typeset=reports,
            stages=stats,
            code_origin=self.origin,
        )
        return result, render.image

    def translate_gt(self, gt: GtPage) -> PageResult:
        """Translate ground-truth source texts (reading order) with the configured providers."""
        from manga_ar.schemas import BBox, OcrResult, Region, RegionType

        self._seed()
        # v0.1.0 cannot declare other sources; its undetermined-language default is ja.
        src = gt.lang if gt.lang in SOURCE_LANGS else "ja"
        regions = []
        for g in sorted(gt.regions, key=lambda r: r.reading_order):
            if g.type == "sfx":
                continue
            regions.append(
                Region(
                    id=g.region_id,
                    type=RegionType(PROMPT_TO_V010[g.type]),
                    bbox=BBox(0, 0, 1, 1),
                    reading_order=g.reading_order,
                    source_lang=src,
                    ocr=OcrResult(engine="gt", text=g.text, lang=src),
                )
            )
        stats: dict[str, StageStats] = {}
        with measure(stats, "translate"):
            self.pipeline.stages.translator.translate_regions(regions, src)
        texts = {r.id: r.translation.text for r in regions if r.translation is not None}
        return PageResult(
            page_id=gt.page_id,
            version="",
            mode="translate_gt",
            translations=texts,
            stages=stats,
            code_origin=self.origin,
        )
