"""Stage orchestration: Load → Detect/segment → OCR → Inpaint → Translate → Typeset → Export.

The CLI and the GUI both drive :class:`Pipeline`. Failures are isolated at two levels:
a region that fails OCR, inpainting or typesetting is flagged and the page continues; a
page that fails to load or crashes is recorded in the report and the batch continues
(E9, E18). Every exported page gets a sidecar (``<stem>_ar.mangaar.json``) plus two work
images (decoded original and fully inpainted page) under ``.mangaar/`` so it can be
re-typeset later without detection, OCR, inpainting or translation (``rerender``, GUI).
"""

from __future__ import annotations

import os
import re
import time
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from manga_ar.cancel import CancelToken
from manga_ar.config import AppConfig, build_config, resolve_cache_dir
from manga_ar.detect.base import BubbleSegmenter, TextDetector
from manga_ar.detect.reading_order import assign_reading_order, default_mode
from manga_ar.detect.tiled import detect_page
from manga_ar.errors import CancelledError, ConfigError, ImageLoadError, MangaArError
from manga_ar.io.archive import ArchiveLimits, read_archive, write_cbz
from manga_ar.io.loader import LoadedImage, LoadLimits, load_image_bytes
from manga_ar.io.naming import output_name, sidecar_name
from manga_ar.io.sources import PageJob, collect_inputs
from manga_ar.io.writer import atomic_write_bytes, encode_image, write_image
from manga_ar.logging_setup import get_logger
from manga_ar.models.manager import ModelManager
from manga_ar.ocr.langid import is_passthrough
from manga_ar.report import BatchReport, PageOutcome, PageStatus
from manga_ar.schemas import (
    Flag,
    OcrResult,
    PageDocument,
    Region,
    RegionType,
    TranslationResult,
)
from manga_ar.typeset.fonts import FontRegistry
from manga_ar.typeset.layout import Typesetter
from manga_ar.typeset.page import PageRender, typeset_page

log = get_logger(__name__)
RgbArray = npt.NDArray[np.uint8]
ProgressFn = Callable[[float, str], None]
WORK_DIR = ".mangaar"
_DEGRADING = {Flag.UNTRANSLATED, Flag.TYPESET_FAILED, Flag.OCR_FAILED}


# ------------------------------------------------------------------ stage protocols
class OcrStage(Protocol):
    def recognize(self, page: RgbArray, region: Region, lang: str) -> OcrResult: ...

    def detect_language(
        self, page: RgbArray, regions: Sequence[Region], max_samples: int = 4
    ) -> tuple[str | None, dict[str, float]]: ...


class InpaintStage(Protocol):
    def eligible(self, region: Region) -> bool: ...

    def inpaint_region(self, page: RgbArray, region: Region) -> str: ...


class TranslateStage(Protocol):
    def translate_regions(self, regions: Sequence[Region], src: str) -> None: ...

    def translate_text(self, text: str, src: str) -> TranslationResult | None: ...

    def has_offline_provider(self, src: str) -> bool: ...


@dataclass
class Stages:
    detector: TextDetector
    segmenter: BubbleSegmenter
    ocr: OcrStage
    inpainter: InpaintStage
    translator: TranslateStage
    typesetter: Typesetter


def build_stages(
    cfg: AppConfig, manager: ModelManager, tm_pairs: dict[str, str] | None = None
) -> Stages:
    """Production stages for ``cfg`` (models load lazily on first use)."""
    from manga_ar.detect.bubble import FloodBubbleSegmenter
    from manga_ar.detect.classical import ClassicalDetector
    from manga_ar.detect.factory import build_detector
    from manga_ar.inpaint.lama import LamaInpainter
    from manga_ar.inpaint.residual import ResidualChecker
    from manga_ar.inpaint.strategy import RegionInpainter
    from manga_ar.models.device import resolve_device
    from manga_ar.ocr.factory import build_router
    from manga_ar.translate.service import build_translation_service

    device = resolve_device(cfg.runtime.device)
    lama = (
        LamaInpainter(manager, device, cfg.inpaint.lama_max_side) if cfg.inpaint.use_lama else None
    )
    residual = (
        ResidualChecker(ClassicalDetector(cfg.detect)) if cfg.inpaint.residual_check else None
    )
    return Stages(
        detector=build_detector(cfg, manager),
        segmenter=FloodBubbleSegmenter(cfg.detect),
        ocr=build_router(cfg, manager, device),
        inpainter=RegionInpainter(cfg.inpaint, lama, residual),
        translator=build_translation_service(cfg, manager, resolve_cache_dir(cfg), tm_pairs),
        typesetter=Typesetter(cfg.typeset, FontRegistry()),
    )


# ------------------------------------------------------------------------ paths
@dataclass
class PagePaths:
    sidecar: Path
    clean: Path
    source: Path
    output: Path | None  # None when pages are packed into a CBZ
    member: str | None  # CBZ member name
    cbz: Path | None
    debug_dir: Path

    @classmethod
    def for_job(cls, root: Path, job: PageJob, cfg: AppConfig) -> PagePaths:
        out = cfg.output
        base = root / job.rel_dir
        work = base / WORK_DIR
        if out.format == "cbz":
            member = str(
                PurePosixPath(job.member).parent / output_name(job.stem, "png", out.suffix)
            )
            cbz = root / job.group_dir / f"{job.group_name}{out.suffix}.cbz"
            output = None
        else:
            member, cbz = None, None
            output = base / output_name(job.stem, out.format, out.suffix)
        return cls(
            sidecar=base / sidecar_name(job.stem, out.suffix),
            clean=work / f"{job.stem}.clean.png",
            source=work / f"{job.stem}.source.png",
            output=output,
            member=member.removeprefix("./") if member else None,
            cbz=cbz,
            debug_dir=root / "debug" / job.rel_dir / job.stem,
        )


def _rel(target: Path, start: Path) -> str:
    return Path(os.path.relpath(target, start)).as_posix()


@dataclass
class _LangVote:
    scores: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    pages: int = 0

    def add(self, scores: dict[str, float]) -> None:
        for lang, value in scores.items():
            self.scores[lang] += value
        self.pages += 1

    @property
    def decided(self) -> bool:
        return self.pages >= 3

    def best(self) -> str | None:
        if not self.scores:
            return None
        return max(self.scores, key=lambda k: self.scores[k])


@dataclass
class PageResult:
    """In-memory result of one page (before export)."""

    doc: PageDocument
    clean: RgbArray
    render: PageRender
    inpaint_failures: int = 0

    @property
    def status(self) -> PageStatus:
        if not self.doc.regions:
            return "no_text"
        degraded = self.inpaint_failures or any(r.flags & _DEGRADING for r in self.doc.regions)
        return "degraded" if degraded else "ok"


# --------------------------------------------------------------------- pipeline
class Pipeline:
    def __init__(
        self,
        cfg: AppConfig,
        stages: Stages | None = None,
        *,
        manager: ModelManager | None = None,
        tm_pairs: dict[str, str] | None = None,
        cancel: CancelToken | None = None,
        progress: ProgressFn | None = None,
    ) -> None:
        self.cfg = cfg
        self._stages = stages
        self.manager = manager or ModelManager(resolve_cache_dir(cfg), offline=cfg.runtime.offline)
        self.tm_pairs = tm_pairs
        self.cancel = cancel or CancelToken()
        self.progress = progress
        self._votes: dict[str, _LangVote] = {}
        self._langs_seen: set[str] = set()

    @property
    def stages(self) -> Stages:
        if self._stages is None:
            self._stages = build_stages(self.cfg, self.manager, self.tm_pairs)
        return self._stages

    # -------------------------------------------------------------- batch API
    def run(self, inputs: Sequence[Path], output_dir: Path) -> BatchReport:
        """Translate every page of ``inputs`` into ``output_dir``; writes report.json/.md."""
        started = time.perf_counter()
        cfg = self.cfg
        report = BatchReport(cfg.config_hash(), cfg.preset, str(output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        collected = collect_inputs(inputs, self._archive_limits(), exclude=output_dir)
        for problem in collected.problems:
            status: PageStatus = "failed" if problem.status == "failed" else "skipped"
            report.add(PageOutcome(problem.name, status, error=problem.error))
        jobs = collected.jobs
        books: dict[Path, list[tuple[str, bytes]]] = defaultdict(list)
        total = max(1, len(jobs))
        try:
            for index, job in enumerate(jobs):
                self.cancel.check()
                self._progress(index / total, f"{job.name} ({index + 1}/{len(jobs)})")
                paths = PagePaths.for_job(output_dir, job, cfg)
                try:
                    outcome, member = self.process_job(job, paths)
                except CancelledError:
                    report.add(PageOutcome(job.name, "cancelled"))
                    raise
                report.add(outcome)
                if member is not None and paths.cbz is not None and paths.member is not None:
                    books[paths.cbz].append((paths.member, member))
        except CancelledError:
            report.cancelled = True
            log.warning("cancelled; %d page(s) not processed", len(jobs) - len(report.pages))
        finally:
            for cbz_path, entries in books.items():
                write_cbz(cbz_path, entries)
                report.archives.append(_rel(cbz_path, output_dir))
        self._notes(report)
        report.elapsed_s = time.perf_counter() - started
        report.save(output_dir)
        self._progress(1.0, "done")
        return report

    def process_job(self, job: PageJob, paths: PagePaths) -> tuple[PageOutcome, bytes | None]:
        """Load, (resume or) translate and export one page. Returns the outcome and, for
        CBZ output, the encoded page."""
        try:
            loaded = load_image_bytes(job.read(), job.name, self._load_limits())
        except ImageLoadError as exc:
            log.warning("skipping %s", exc)
            return PageOutcome(job.name, "skipped", error=str(exc)), None
        if self.cfg.output.resume and not self.cfg.output.force:
            resumed = self._try_resume(job, paths, loaded)
            if resumed is not None:
                return resumed
        timings: dict[str, float] = {}
        try:
            result = self.translate_image(loaded, job, timings)
            return self._export(job, paths, loaded, result, timings)
        except CancelledError:
            raise
        except Exception as exc:
            log.error("%s: page failed (%s: %s)", job.name, type(exc).__name__, exc)
            log.debug("traceback", exc_info=True)
            return (
                PageOutcome(
                    job.name, "failed", error=f"{type(exc).__name__}: {exc}", timings=timings
                ),
                None,
            )

    # ------------------------------------------------------------- page core
    def translate_image(
        self, loaded: LoadedImage, job: PageJob, timings: dict[str, float] | None = None
    ) -> PageResult:
        """All stages on one decoded page, in memory."""
        timings = timings if timings is not None else {}
        cfg, stages, rgb = self.cfg, self.stages, loaded.rgb
        doc = PageDocument(
            source=job.name,
            width=loaded.width,
            height=loaded.height,
            source_sha256=loaded.sha256,
            config_hash=cfg.config_hash(),
            was_grayscale=loaded.was_grayscale,
            warnings=list(loaded.warnings),
        )
        page_id = re.sub(r"[^\w.-]", "_", job.stem) or "page"
        with _timed(timings, "detect"):
            blocks = detect_page(rgb, stages.detector, cfg.tiling)
            regions = stages.segmenter.segment(rgb, blocks, page_id)
        self.cancel.check()
        if not regions:
            log.info("%s: no text detected; exported unchanged", job.name)
            doc.timings = timings
            empty = np.zeros((*rgb.shape[:2], 4), np.uint8)
            return PageResult(doc, rgb, PageRender(rgb.copy(), empty))
        if cfg.detect.sfx == "translate":
            for r in regions:
                if r.type == RegionType.SFX:
                    r.type = RegionType.FREE_TEXT
                    r.flag(Flag.FROM_SFX)
        active = [r for r in regions if r.type != RegionType.SFX]
        sfx = [r for r in regions if r.type == RegionType.SFX]
        lang = self._language(rgb, active, job.group, doc, timings)
        doc.lang = lang
        mode = cfg.input.reading_order if cfg.input.reading_order != "auto" else default_mode(lang)
        doc.reading_order_mode = mode
        active = assign_reading_order(active, rgb, mode) if active else []
        for k, r in enumerate(sfx):
            r.reading_order = len(active) + k
        doc.regions = [*active, *sfx]
        with _timed(timings, "ocr"):
            self._ocr(rgb, active, lang, doc)
        with _timed(timings, "inpaint"):
            clean, failures = self._inpaint(rgb, active, doc)
        with _timed(timings, "translate"):
            self._translate(active, lang, doc)
        self.cancel.check()
        with _timed(timings, "typeset"):
            render = self._typeset(doc, rgb, clean)
        doc.timings = timings
        return PageResult(doc, clean, render, failures)

    def _language(
        self,
        rgb: RgbArray,
        regions: list[Region],
        group: str,
        doc: PageDocument,
        timings: dict[str, float],
    ) -> str:
        """Explicit ``--source``, else a document-level vote over the first pages of each
        book (E6): scores accumulate until three pages have voted."""
        src = self.cfg.input.source_lang
        if src != "auto":
            return src
        vote = self._votes.setdefault(group, _LangVote())
        if not vote.decided and regions:
            with _timed(timings, "langid"):
                lang, scores = self.stages.ocr.detect_language(rgb, regions)
            if lang is not None:
                vote.add(scores)
        best = vote.best()
        if best is None:
            doc.warnings.append("source language undetermined; assuming ja (use --source)")
            return "ja"
        return best

    def _ocr(self, rgb: RgbArray, regions: list[Region], lang: str, doc: PageDocument) -> None:
        for region in regions:
            self.cancel.check()
            if region.override.skip:
                continue
            region_lang = region.override.source_lang or lang
            region.source_lang = region_lang
            self._langs_seen.add(region_lang)
            try:
                region.ocr = self.stages.ocr.recognize(rgb, region, region_lang)
            except Exception as exc:  # noqa: BLE001 - region isolation
                log.warning("%s: OCR crashed (%s: %s)", region.id, type(exc).__name__, exc)
                region.flag(Flag.OCR_FAILED)
                doc.warnings.append(f"{region.id}: OCR failed ({type(exc).__name__})")
                continue
            text = region.ocr.text
            if self.cfg.ocr.pass_through_latin and text and is_passthrough(text):
                region.flag(Flag.PASS_THROUGH)

    def _inpaint(
        self, rgb: RgbArray, regions: list[Region], doc: PageDocument
    ) -> tuple[RgbArray, int]:
        clean = rgb.copy()
        failures = 0
        inpainter = self.stages.inpainter
        for region in regions:
            self.cancel.check()
            if not inpainter.eligible(region):
                continue
            try:
                inpainter.inpaint_region(clean, region)
            except Exception as exc:  # noqa: BLE001 - region isolation
                # A partial pass may have written inside region.inpaint_mask; the region
                # is skipped, so composition restores its original pixels there.
                failures += 1
                region.flag(Flag.SKIPPED)
                doc.warnings.append(f"{region.id}: inpainting failed ({exc}); left untouched")
                log.warning("%s: inpainting failed (%s)", region.id, exc)
        return clean, failures

    def _translate(self, regions: list[Region], lang: str, doc: PageDocument) -> None:
        todo = [r for r in regions if Flag.SKIPPED not in r.flags]
        by_lang: dict[str, list[Region]] = defaultdict(list)
        for r in todo:
            by_lang[r.source_lang or lang].append(r)
        for src, group in by_lang.items():
            try:
                self.stages.translator.translate_regions(group, src)
            except Exception as exc:  # noqa: BLE001 - never lose the page to translation
                log.warning("translation failed for %d region(s): %s", len(group), exc)
                doc.warnings.append(f"translation failed ({type(exc).__name__}: {exc})")
                for r in group:
                    if r.translation is None and r.override.text is None:
                        r.flag(Flag.UNTRANSLATED)

    def _typeset(self, doc: PageDocument, original: RgbArray, clean: RgbArray) -> PageRender:
        return typeset_page(
            doc,
            original,
            clean,
            self.stages.typesetter,
            erase_untranslated=self.cfg.inpaint.erase_untranslated,
            shadow=self.cfg.typeset.shadow,
        )

    # ---------------------------------------------------------------- export
    def _export(
        self,
        job: PageJob,
        paths: PagePaths,
        loaded: LoadedImage,
        result: PageResult,
        timings: dict[str, float],
    ) -> tuple[PageOutcome, bytes | None]:
        cfg = self.cfg
        doc = result.doc
        with _timed(timings, "export"):
            data = self._encode(result.render.image, loaded.icc_profile, loaded.was_grayscale)
            if paths.output is not None:
                atomic_write_bytes(paths.output, data)
            doc.status = result.status
            doc.timings = timings
            if cfg.output.write_sidecar:
                # Work images favour speed (lossless either way; ~3x faster to write).
                write_image(paths.clean, result.clean, "png", png_compress_level=1)
                write_image(paths.source, loaded.rgb, "png", png_compress_level=1)
                self._describe_outputs(doc, paths)
                doc.save(paths.sidecar)
            if cfg.runtime.debug:
                self._write_debug(paths.debug_dir, loaded.rgb, result)
        outcome = self._outcome(job.name, doc, paths, result.status)
        return outcome, data if paths.member is not None else None

    def _encode(self, image: RgbArray, icc: bytes | None, grayscale: bool) -> bytes:
        out = self.cfg.output
        fmt = "png" if out.format == "cbz" else out.format
        return encode_image(
            image,
            fmt,
            jpeg_quality=out.jpeg_quality,
            webp_quality=out.webp_quality,
            icc_profile=icc if out.keep_icc else None,
            grayscale=grayscale,
        )

    def _describe_outputs(self, doc: PageDocument, paths: PagePaths) -> None:
        base = paths.sidecar.parent
        doc.clean_image = _rel(paths.clean, base)
        doc.source_image = _rel(paths.source, base)
        target = paths.output if paths.output is not None else paths.cbz
        doc.output = _rel(target, base) if target is not None else None
        doc.output_member = paths.member
        doc.settings = self.cfg.to_dict()

    def _outcome(
        self, name: str, doc: PageDocument, paths: PagePaths, status: PageStatus
    ) -> PageOutcome:
        flags: dict[str, int] = defaultdict(int)
        for r in doc.regions:
            for f in r.flags:
                flags[f.value] += 1
        target = paths.output or paths.cbz
        return PageOutcome(
            name,
            status,
            output=str(target) + (f"!{paths.member}" if paths.member else "") if target else None,
            sidecar=str(paths.sidecar) if self.cfg.output.write_sidecar else None,
            lang=doc.lang,
            regions=len(doc.regions),
            translated=sum(r.layout is not None for r in doc.regions),
            flags=dict(flags),
            warnings=list(doc.warnings),
            timings=dict(doc.timings),
        )

    def _write_debug(self, directory: Path, rgb: RgbArray, result: PageResult) -> None:
        from PIL import Image

        from manga_ar.debug import (
            detection_overlay,
            inpaint_diff,
            layout_overlay,
            mask_overlay,
            write_debug,
        )

        regions = result.doc.regions
        write_debug(directory, "01_detections", detection_overlay(rgb, regions))
        write_debug(directory, "02_text_masks", mask_overlay(rgb, regions, "text"))
        write_debug(directory, "03_inpaint_masks", mask_overlay(rgb, regions, "inpaint"))
        write_debug(directory, "04_inpaint_diff", inpaint_diff(rgb, result.clean))
        write_debug(directory, "05_clean", result.clean)
        write_debug(directory, "06_layout", layout_overlay(result.render.image, regions))
        import io

        buf = io.BytesIO()
        Image.fromarray(result.render.layer, "RGBA").save(buf, "PNG")
        atomic_write_bytes(directory / "07_text_layer.png", buf.getvalue())

    # ---------------------------------------------------------------- resume
    def _try_resume(
        self, job: PageJob, paths: PagePaths, loaded: LoadedImage
    ) -> tuple[PageOutcome, bytes | None] | None:
        """Skip pages already done with the same config (E19). Degraded pages retry the
        translation of their untranslated regions from the sidecar, then re-render."""
        if not paths.sidecar.is_file():
            return None
        try:
            doc = PageDocument.load(paths.sidecar)
        except MangaArError as exc:
            log.info("%s: unreadable sidecar (%s); reprocessing", job.name, exc)
            return None
        if doc.config_hash != self.cfg.config_hash() or doc.source_sha256 != loaded.sha256:
            return None
        if doc.status not in {"ok", "no_text", "degraded"}:
            return None
        complete = doc.status in {"ok", "no_text"}
        if complete and paths.output is not None and paths.output.is_file():
            outcome = self._outcome(job.name, doc, paths, "resumed")
            return outcome, None
        try:
            original, clean = load_work_images(doc, paths.sidecar)
        except MangaArError as exc:
            log.info("%s: work images missing (%s); reprocessing", job.name, exc)
            return None
        if doc.status == "degraded":
            self._retry_untranslated(doc)
        render = self._typeset(doc, original, clean)
        result = PageResult(doc, clean, render)
        data = self._encode(render.image, loaded.icc_profile, loaded.was_grayscale)
        if paths.output is not None:
            atomic_write_bytes(paths.output, data)
        doc.status = result.status
        doc.save(paths.sidecar)
        status: PageStatus = "resumed" if doc.status != "degraded" else "degraded"
        return self._outcome(job.name, doc, paths, status), (
            data if paths.member is not None else None
        )

    def _retry_untranslated(self, doc: PageDocument) -> None:
        retry = [
            r
            for r in doc.regions
            if Flag.UNTRANSLATED in r.flags and r.ocr is not None and r.ocr.text
        ]
        for r in retry:
            r.flags.discard(Flag.UNTRANSLATED)
        if retry:
            self._translate(retry, doc.lang or "ja", doc)

    # ------------------------------------------------------------- utilities
    def _progress(self, fraction: float, message: str) -> None:
        if self.progress is not None:
            self.progress(fraction, message)

    def _load_limits(self) -> LoadLimits:
        i = self.cfg.input
        return LoadLimits(i.max_pixels, i.min_side, i.max_side, i.truncated)

    def _archive_limits(self) -> ArchiveLimits:
        i = self.cfg.input
        mb = 1024 * 1024
        return ArchiveLimits(
            i.archive_max_members, i.archive_max_member_mb * mb, i.archive_max_total_mb * mb
        )

    def _notes(self, report: BatchReport) -> None:
        untranslated = sum(p.flags.get(Flag.UNTRANSLATED.value, 0) for p in report.pages)
        if not untranslated:
            return
        report.note(
            f"{untranslated} region(s) stayed UNTRANSLATED (original pixels kept unless "
            "--erase-untranslated); edit them in the GUI or rerun with --resume."
        )
        if self._stages is None:
            return
        offline_ok = all(
            self._stages.translator.has_offline_provider(lang) for lang in self._langs_seen
        )
        if not offline_ok:
            report.note(
                "No offline translation model is installed. While online, run "
                "`manga-arabic models download local-mt` (Marian ja/ko/zh→en + en→ar, "
                "CC-BY-4.0) to translate without network access."
            )


@contextmanager
def _timed(timings: dict[str, float], key: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[key] = timings.get(key, 0.0) + time.perf_counter() - start


# --------------------------------------------------------------------- rerender
def load_work_images(doc: PageDocument, sidecar: Path) -> tuple[RgbArray, RgbArray]:
    """(original, clean) pages referenced by a sidecar."""
    from manga_ar.io.loader import load_image_file

    if not doc.clean_image or not doc.source_image:
        raise MangaArError(f"{sidecar.name}: sidecar has no work images (cannot rerender)")
    base = sidecar.parent
    original = load_image_file(base / doc.source_image).rgb
    clean = load_image_file(base / doc.clean_image).rgb
    if original.shape != clean.shape or original.shape[:2] != (doc.height, doc.width):
        raise MangaArError(f"{sidecar.name}: work images do not match the page size")
    return original, clean


def settings_config(doc: PageDocument, fallback: AppConfig, overrides: dict[str, Any]) -> AppConfig:
    """The configuration a page was produced with, plus explicit overrides."""
    cfg = fallback
    if doc.settings:
        try:
            cfg = build_config(doc.settings)
        except ConfigError as exc:
            log.warning("sidecar settings unusable (%s); using the current configuration", exc)
    clean = {k: v for k, v in overrides.items() if v is not None}
    return cfg.replace(**clean) if clean else cfg


@dataclass
class RerenderResult:
    doc: PageDocument
    image: RgbArray
    output: Path


def rerender(
    sidecar: Path,
    fallback: AppConfig,
    overrides: dict[str, Any] | None = None,
    registry: FontRegistry | None = None,
    doc: PageDocument | None = None,
) -> RerenderResult:
    """Re-typeset a page from its sidecar (user edits, font change) and re-export it.

    Only typesetting runs: detection, OCR, inpainting and translation are reused (A14).
    ``doc`` may carry in-memory edits (GUI); it is saved back to ``sidecar``."""
    doc = doc or PageDocument.load(sidecar)
    cfg = settings_config(doc, fallback, overrides or {})
    original, clean = load_work_images(doc, sidecar)
    ts = Typesetter(cfg.typeset, registry or FontRegistry())
    render = typeset_page(
        doc,
        original,
        clean,
        ts,
        erase_untranslated=cfg.inpaint.erase_untranslated,
        shadow=cfg.typeset.shadow,
    )
    if not doc.output:
        raise MangaArError(f"{sidecar.name}: sidecar has no output path")
    target = sidecar.parent / doc.output
    out = cfg.output
    fmt = "png" if doc.output_member else (target.suffix.lstrip(".").lower() or out.format)
    data = encode_image(
        render.image,
        fmt,
        jpeg_quality=out.jpeg_quality,
        webp_quality=out.webp_quality,
        grayscale=doc.was_grayscale,
    )
    if doc.output_member:
        _replace_cbz_member(target, doc.output_member, data)
    else:
        atomic_write_bytes(target, data)
    failed = any(r.flags & _DEGRADING for r in doc.regions)
    doc.status = "degraded" if failed else ("ok" if doc.regions else "no_text")
    doc.settings = cfg.to_dict()
    doc.save(sidecar)
    return RerenderResult(doc, render.image, target)


def _replace_cbz_member(path: Path, member: str, data: bytes) -> None:
    entries: list[tuple[str, bytes]] = []
    if path.is_file():
        contents = read_archive(path)
        entries = [(m.name, m.data) for m in contents.members if m.name != member]
    entries.append((member, data))
    from manga_ar.io.naming import natural_key

    entries.sort(key=lambda e: natural_key(e[0]))
    write_cbz(path, entries)
