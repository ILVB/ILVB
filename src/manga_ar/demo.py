"""``manga-arabic demo``: installation smoke test on a generated page (no network needed
once the CJK fixture font is cached).

A fixed Chinese page is drawn with an OFL Noto Sans SC subset, then the real pipeline
runs on it: classical detection, RapidOCR, solid-fill/OpenCV inpainting, translation
through a translation memory holding the three demo lines (offline), Arabic typesetting
and export. Success means every bubble came out translated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from manga_ar import synth
from manga_ar.config import AppConfig, resolve_cache_dir
from manga_ar.errors import ModelUnavailableError
from manga_ar.io.writer import write_image
from manga_ar.logging_setup import get_logger
from manga_ar.models.manager import ModelManager
from manga_ar.pipeline import Pipeline, ProgressFn
from manga_ar.report import EXIT_FATAL, EXIT_OK, EXIT_PARTIAL, BatchReport

log = get_logger(__name__)

DEMO_OVERRIDES: dict[str, Any] = {
    "input.source_lang": "zh",
    "translate.providers": ["tm"],
    "translate.cache": False,
    "inpaint.use_lama": False,  # uniform bubbles: solid fill; no 200 MB download needed
    "runtime.offline": True,
    "output.format": "png",
    "output.resume": False,
}


@dataclass
class DemoResult:
    input_path: Path
    output_path: Path | None
    report: BatchReport
    expected: int
    translated: int

    @property
    def exit_code(self) -> int:
        if self.report.exit_code == EXIT_FATAL or self.translated == 0:
            return EXIT_FATAL
        return EXIT_OK if self.translated == self.expected else EXIT_PARTIAL


def ensure_demo_font(cfg: AppConfig) -> Path:
    cache = resolve_cache_dir(cfg)
    try:
        return synth.find_cjk_font("zh", cache)
    except ModelUnavailableError:
        if cfg.runtime.offline:
            raise
    log.info("downloading the Noto Sans SC subset (OFL) used to draw the demo page")
    ModelManager(cache).ensure("font-cjk-sc")
    return synth.find_cjk_font("zh", cache)


def run_demo(cfg: AppConfig, output_dir: Path, progress: ProgressFn | None = None) -> DemoResult:
    ensure_demo_font(cfg)
    page = synth.demo_page(resolve_cache_dir(cfg))
    input_path = output_dir / "input" / f"{page.name}.png"
    write_image(input_path, page.image, "png")
    demo_cfg = cfg.replace(**DEMO_OVERRIDES)
    pipeline = Pipeline(demo_cfg, tm_pairs=dict(synth.DEMO_TRANSLATIONS), progress=progress)
    from manga_ar.ocr.router import OcrRouter

    router = pipeline.stages.ocr
    if isinstance(router, OcrRouter) and not router.engines_for("zh"):
        raise ModelUnavailableError(
            "no Chinese OCR engine is available; install the 'rapid' extra "
            '(pip install "manga-arabic-translator[rapid]")'
        )
    report = pipeline.run([input_path], output_dir / "output")
    page_out = report.pages[0] if report.pages else None
    translated = page_out.translated if page_out is not None else 0
    output = Path(page_out.output) if page_out is not None and page_out.output else None
    return DemoResult(input_path, output, report, len(page.regions), translated)
