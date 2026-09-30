"""Bubble segmentation gate (P2): mean mask IoU ≥ 0.85, leak fallback, typing, merge."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from manga_ar import synth
from manga_ar.config import load_config
from manga_ar.detect.bubble import FloodBubbleSegmenter, inscribed_safe_box
from manga_ar.detect.classical import ClassicalDetector
from manga_ar.detect.geometry import fill_holes, largest_inscribed_rect, solidity
from manga_ar.metrics import mask_iou, match_boxes
from manga_ar.schemas import Flag, RegionType

CFG = load_config(environ={})


def _segment(page: synth.SynthPage) -> list:  # type: ignore[type-arg]
    blocks = ClassicalDetector(CFG.detect).detect(page.image)
    return FloodBubbleSegmenter(CFG.detect).segment(page.image, blocks, page.name)


def test_mask_iou_and_types(cjk_ready: Path, cache_dir: Path) -> None:
    pages = [synth.basic_page(lang, 30, cache_dir) for lang in ("ja", "ko", "zh")]
    pages += [synth.variety_page(lang, 31, cache_dir) for lang in ("ja", "ko", "zh")]
    ious, types_ok, n = [], 0, 0
    for page in pages:
        h, w = page.image.shape[:2]
        regions = [r for r in _segment(page) if r.type != RegionType.SFX]
        for gi, rj in match_boxes(
            [g.text_bbox for g in page.regions], [r.bbox for r in regions], 0.5
        ):
            g, r = page.regions[gi], regions[rj]
            n += 1
            types_ok += g.type == r.type.value
            if g.bubble_mask is not None and r.bubble_mask is not None:
                ious.append(mask_iou(g.bubble_mask.to_full(h, w), r.bubble_mask.to_full(h, w)))
    assert float(np.mean(ious)) >= 0.85, np.mean(ious)
    assert types_ok / n >= 0.95, (types_ok, n)


def test_leak_fallback_on_open_bubble(cjk_ready: Path, cache_dir: Path) -> None:
    page = synth.adversarial_gutter_page(0, cache_dir)
    (region,) = _segment(page)
    assert Flag.LEAK_FALLBACK in region.flags and region.type == RegionType.BUBBLE
    h, w = page.image.shape[:2]
    area = region.bubble_mask.area  # type: ignore[union-attr]
    assert area < 0.1 * h * w  # the fallback shape is local, not the leaked gutter
    assert region.safe_box is not None and region.safe_box.contains(region.bbox.expand(-2))


def test_two_blocks_in_one_bubble_are_merged(cjk_ready: Path, cache_dir: Path) -> None:
    page = synth.two_block_bubble_page(0, cache_dir)
    regions = _segment(page)
    assert len(regions) == 1
    assert Flag.MERGED in regions[0].flags
    assert regions[0].bbox.iou(page.regions[0].text_bbox) > 0.9
    assert regions[0].lines == sorted(regions[0].lines, key=lambda b: -b.x1)  # right → left


def test_free_text_and_narration(cjk_ready: Path, cache_dir: Path) -> None:
    page = synth.variety_page("ja", 0, cache_dir)
    kinds = {r.type for r in _segment(page)}
    assert {RegionType.FREE_TEXT, RegionType.NARRATION, RegionType.SFX} <= kinds


def _brute_rect_area(mask: np.ndarray) -> int:
    h = mask.shape[0]
    best = 0
    for y0 in range(h):
        for y1 in range(y0 + 1, h + 1):
            rows = mask[y0:y1].all(axis=0)
            run = cur = 0
            for v in rows:
                cur = cur + 1 if v else 0
                run = max(run, cur)
            best = max(best, run * (y1 - y0))
    return best


@settings(max_examples=40, deadline=None)
@given(arrays(np.bool_, st.tuples(st.integers(1, 9), st.integers(1, 9))))
def test_largest_inscribed_rect_matches_brute_force(mask: np.ndarray) -> None:
    rect = largest_inscribed_rect(mask)
    expected = _brute_rect_area(mask)
    if expected == 0:
        assert rect is None
    else:
        assert rect is not None and rect.area == expected
        assert mask[rect.y0 : rect.y1, rect.x0 : rect.x1].all()


def test_geometry_helpers() -> None:
    ring = np.zeros((30, 30), bool)
    ring[5:25, 5:25] = True
    ring[10:20, 10:20] = False
    assert fill_holes(ring)[15, 15]
    assert solidity(ring) > 0.95
    star = np.zeros((40, 40), bool)
    star[18:22, :] = True
    star[:, 18:22] = True
    assert solidity(star) < 0.5
    safe = inscribed_safe_box(ring | fill_holes(ring), (100, 200), erode=2)
    assert safe is not None and safe.x0 >= 107 and safe.y0 >= 207
