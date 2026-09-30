"""Detection gate (P2): recall ≥ 0.95, precision ≥ 0.90 @ IoU 0.5 on synthetic pages."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from manga_ar import synth
from manga_ar.config import load_config
from manga_ar.detect.classical import ClassicalDetector
from manga_ar.detect.factory import build_detector
from manga_ar.detect.tiled import detect_page
from manga_ar.metrics import match_boxes, precision_recall
from manga_ar.models.manager import ModelManager

CFG = load_config(environ={})


@pytest.fixture(scope="module")
def pages(cjk_ready: Path, cache_dir: Path) -> list[synth.SynthPage]:
    out = []
    for seed in (20, 21):
        out += [synth.basic_page(lang, seed, cache_dir) for lang in ("ja", "ko", "zh")]
        out.append(synth.basic_page("ja", seed, cache_dir, vertical=False))
        out.append(synth.variety_page(("ja", "ko", "zh")[seed % 3], seed, cache_dir))
    out += [
        synth.furigana_page(0, cache_dir),
        synth.symbols_page("ja", 0, cache_dir),
        synth.symbols_page("ko", 1, cache_dir),
        synth.demo_page(cache_dir),
    ]
    return out


@pytest.fixture(scope="module")
def detector() -> ClassicalDetector:
    return ClassicalDetector(CFG.detect)


def test_recall_precision_gate(pages: list[synth.SynthPage], detector: ClassicalDetector) -> None:
    n_t = n_p = n_m = 0
    for page in pages:
        blocks = [b for b in detector.detect(page.image) if not b.is_sfx]
        m = match_boxes([g.text_bbox for g in page.regions], [b.bbox for b in blocks], 0.5)
        n_t, n_p, n_m = n_t + len(page.regions), n_p + len(blocks), n_m + len(m)
    precision, recall = precision_recall(n_t, n_p, n_m)
    assert recall >= 0.95, (recall, n_t, n_m)
    assert precision >= 0.90, (precision, n_p, n_m)


def test_orientation_matches_ground_truth(
    pages: list[synth.SynthPage], detector: ClassicalDetector
) -> None:
    total = correct = 0
    for page in pages:
        blocks = detector.detect(page.image)
        for gi, bj in match_boxes(
            [g.text_bbox for g in page.regions], [b.bbox for b in blocks], 0.5
        ):
            if len(page.regions[gi].text) >= 3:
                total += 1
                correct += blocks[bj].vertical == page.regions[gi].vertical
    assert correct / total >= 0.97, (correct, total)


def test_sfx_flagged_and_text_masks_exact(
    cjk_ready: Path, cache_dir: Path, detector: ClassicalDetector
) -> None:
    page = synth.variety_page("ja", 0, cache_dir)
    blocks = detector.detect(page.image)
    assert any(b.is_sfx for b in blocks)
    h, w = page.image.shape[:2]
    for g in page.regions:
        best = max(blocks, key=lambda b: b.bbox.iou(g.text_bbox))
        pred = best.text_mask.to_full(h, w)  # type: ignore[union-attr]
        truth = g.text_mask.to_full(h, w)
        covered = (pred & truth).sum() / truth.sum()
        assert covered >= 0.9, (g.text, covered)  # glyph pixels are found for inpainting


def test_furigana_excluded_from_lines(
    cjk_ready: Path, cache_dir: Path, detector: ClassicalDetector
) -> None:
    page = synth.furigana_page(0, cache_dir)
    blocks = detector.detect(page.image)
    assert len(blocks) == 2
    for b in blocks:
        assert b.furigana, "furigana column not identified"
        assert all(ln.width > 0.6 * b.glyph_size for ln in b.lines)


def test_negative_controls_no_text() -> None:
    det = ClassicalDetector(CFG.detect)
    blank = np.full((600, 500, 3), 255, np.uint8)
    assert det.detect(blank) == []
    art = blank.copy()
    for y in range(0, 600, 6):  # screentone
        art[y : y + 2, ::6] = 90
    for k in range(-600, 500, 14):  # hatching
        for t in range(600):
            x = k + t
            if 0 <= x < 500:
                art[t, x] = 40
    assert [b for b in det.detect(art) if not b.is_sfx] == []


def test_webtoon_seam_each_bubble_once(cjk_ready: Path, cache_dir: Path) -> None:
    page = synth.webtoon_strip(0, cache_dir)
    blocks = detect_page(page.image, ClassicalDetector(CFG.detect), CFG.tiling)
    for g in page.regions:
        hits = [b for b in blocks if b.bbox.iou(g.text_bbox) >= 0.5]
        assert len(hits) == 1, (g.text, g.text_bbox, [b.bbox for b in hits])
    assert len([b for b in blocks if not b.is_sfx]) == len(page.regions)


def test_factory_falls_back_to_classical(tmp_path: Path) -> None:
    cfg = load_config(overrides={"detect.detector": "ctd", "runtime.offline": True}, environ={})
    det = build_detector(cfg, ModelManager(tmp_path, offline=True))
    assert isinstance(det, ClassicalDetector)
