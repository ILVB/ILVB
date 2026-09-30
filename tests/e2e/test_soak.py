"""50-page soak test (Phase 7): real stages, memory growth and per-page timings.

Chinese synthetic pages (RapidOCR, bundled models) with a translation memory built from
the ground truth, so every page runs detection → OCR → inpainting → translation →
typesetting → export. Resident memory is sampled after every page; after a warm-up the
growth over the remaining pages must stay small (no per-page leak).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from manga_ar import synth
from manga_ar.config import load_config
from manga_ar.io.writer import write_image
from manga_ar.pipeline import Pipeline

pytestmark = pytest.mark.slow
pytest.importorskip("rapidocr_onnxruntime")
PAGES = 50


def _rss_mb() -> float:
    try:
        pages = int(Path("/proc/self/statm").read_text(encoding="ascii").split()[1])
    except OSError:
        pytest.skip("RSS sampling needs /proc (Linux)")
    return pages * os.sysconf("SC_PAGE_SIZE") / 2**20


def test_soak_50_pages(tmp_path: Path, cache_dir: Path) -> None:
    try:
        synth.find_cjk_font("zh", cache_dir)
    except Exception as exc:  # noqa: BLE001 - skip without the fixture font
        pytest.skip(f"CJK font unavailable: {exc}")
    tm: dict[str, str] = {}
    for k in range(PAGES):
        page = (
            synth.basic_page("zh", 1000 + k, cache_dir, vertical=False)
            if k % 2
            else synth.variety_page("zh", 1000 + k, cache_dir)
        )
        for n, region in enumerate(page.regions):
            tm[region.text] = f"نص تجريبي رقم {n + 1}"
        write_image(tmp_path / "in" / f"p{k:03d}.png", page.image, "png")
    cfg = load_config(
        overrides={
            "preset": "fast",
            "input.source_lang": "zh",
            "translate.providers": ["tm"],
            "translate.cache": False,
            "runtime.offline": True,
        },
        environ={},
    )
    samples: list[float] = []
    stamps: list[float] = []

    def progress(fraction: float, message: str) -> None:
        samples.append(_rss_mb())
        stamps.append(time.perf_counter())

    report = Pipeline(cfg, tm_pairs=tm, progress=progress).run([tmp_path / "in"], tmp_path / "out")
    samples.append(_rss_mb())
    ok = [p for p in report.pages if p.succeeded]
    assert len(ok) == PAGES, report.counts
    translated = sum(p.translated for p in ok) / max(1, sum(p.regions for p in ok))
    warm = samples[10]
    growth = max(samples[10:]) - warm
    per_page = (stamps[-1] - stamps[0]) / (len(stamps) - 1)
    print(
        f"\nsoak: {PAGES} pages, translated {translated:.1%} of regions, "
        f"{per_page:.2f} s/page, RSS warm {warm:.0f} MB → peak {max(samples):.0f} MB "
        f"(growth after warm-up {growth:.0f} MB)"
    )
    assert growth < 150, samples  # no per-page leak (50 pages × 3 MB would already show)
    assert translated > 0.8
