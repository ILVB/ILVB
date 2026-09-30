"""Real OCR engines on synthetic pages (P2 gate: CER ≤ 10 % per language/orientation).

Run with: pytest -m integration tests/integration/test_ocr_engines.py
EasyOCR weights download from GitHub on first use; RapidOCR ships its models.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from manga_ar import synth
from manga_ar.config import load_config
from manga_ar.detect.bubble import FloodBubbleSegmenter
from manga_ar.detect.classical import ClassicalDetector
from manga_ar.errors import ModelUnavailableError
from manga_ar.metrics import cer, match_boxes
from manga_ar.models.manager import ModelManager
from manga_ar.ocr.factory import build_router
from manga_ar.schemas import RegionType

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def router(cache_dir: Path):  # type: ignore[no-untyped-def]
    if importlib.util.find_spec("easyocr") is None:
        pytest.skip("easyocr not installed")
    cfg = load_config(environ={"MANGAAR_CACHE_DIR": str(cache_dir)})
    return build_router(cfg, ModelManager(cache_dir), "cpu")


def _page_cer(router, page: synth.SynthPage, lang: str) -> list[float]:  # type: ignore[no-untyped-def]
    cfg = load_config(environ={})
    blocks = ClassicalDetector(cfg.detect).detect(page.image)
    regions = [
        r
        for r in FloodBubbleSegmenter(cfg.detect).segment(page.image, blocks, page.name)
        if r.type != RegionType.SFX
    ]
    router.recognize_all(page.image, regions, lang)
    out = []
    for gi, rj in match_boxes([g.text_bbox for g in page.regions], [r.bbox for r in regions]):
        out.append(cer(page.regions[gi].text, regions[rj].ocr.text))
    return out


@pytest.mark.parametrize(
    ("lang", "vertical"), [("ja", True), ("ja", False), ("ko", False), ("zh", False)]
)
def test_cer_gate(router, cjk_ready: Path, cache_dir: Path, lang: str, vertical: bool) -> None:  # type: ignore[no-untyped-def]
    errors: list[float] = []
    for seed in (50, 51, 52):
        page = synth.basic_page(lang, seed, cache_dir, vertical=None if lang != "ja" else vertical)
        errors += _page_cer(router, page, lang)
    assert len(errors) >= 10
    assert float(np.mean(errors)) <= 0.10, (lang, vertical, np.mean(errors))


def test_language_detection(router, cjk_ready: Path, cache_dir: Path) -> None:  # type: ignore[no-untyped-def]
    cfg = load_config(environ={})
    det, seg = ClassicalDetector(cfg.detect), FloodBubbleSegmenter(cfg.detect)
    for lang in ("ja", "ko", "zh"):
        page = synth.basic_page(lang, 60, cache_dir)
        regions = seg.segment(page.image, det.detect(page.image), page.name)
        found, scores = router.detect_language(page.image, regions)
        assert found == lang, (lang, scores)


def test_manga_ocr_when_hub_reachable(cache_dir: Path) -> None:
    from manga_ar.ocr.manga_ocr_engine import MangaOcrEngine

    engine = MangaOcrEngine(ModelManager(cache_dir))
    page = synth.basic_page("ja", 0, cache_dir)
    g = page.regions[0]
    b = g.text_bbox.expand(10)
    try:
        res = engine.recognize_block(page.image[b.y0 : b.y1, b.x0 : b.x1], "ja", True)
    except ModelUnavailableError as exc:
        pytest.skip(f"manga-ocr weights unavailable here (DECISIONS W-001): {exc}")
    assert cer(g.text, res.text) <= 0.2
