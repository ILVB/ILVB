"""Real LaMa on textured free text (downloads big-lama from GitHub on first use)."""

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
from manga_ar.inpaint.lama import LamaInpainter
from manga_ar.inpaint.strategy import RegionInpainter
from manga_ar.models.manager import ModelManager

pytestmark = pytest.mark.integration


def test_lama_on_texture(cjk_ready: Path, cache_dir: Path) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not installed")
    cfg = load_config(environ={})
    lama = LamaInpainter(ModelManager(cache_dir))
    try:
        lama.inpaint(np.full((64, 64, 3), 200, np.uint8), np.ones((64, 64), bool))
    except ModelUnavailableError as exc:
        pytest.skip(f"LaMa weights unavailable: {exc}")
    page = synth.texture_page(0, cache_dir)
    regions = FloodBubbleSegmenter(cfg.detect).segment(
        page.image, ClassicalDetector(cfg.detect).detect(page.image), page.name
    )
    clean = RegionInpainter(cfg.inpaint, lama).inpaint_page(page.image, regions)
    assert any(r.inpaint_method == "lama" for r in regions)
    h, w = page.image.shape[:2]
    union = np.zeros((h, w), bool)
    for r in regions:
        if r.inpaint_mask is not None:
            union |= r.inpaint_mask.to_full(h, w)
    assert not ((clean != page.image).any(axis=2) & ~union).any()
