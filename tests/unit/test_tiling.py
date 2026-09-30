from __future__ import annotations

import itertools
from pathlib import Path

import cv2
import numpy as np
import pytest

from manga_ar.io.tiling import Tile, is_complete_in_tile, merge_tiled, needs_tiling, plan_tiles
from manga_ar.schemas import BBox


def test_needs_tiling() -> None:
    assert needs_tiling(6000, 720, 2.5, 4096)
    assert needs_tiling(2000, 700, 2.5, 4096)
    assert not needs_tiling(1400, 1000, 2.5, 4096)


@pytest.mark.parametrize("height", [2048, 2049, 5000, 6400, 12345])
def test_plan_tiles_cover_and_overlap(height: int) -> None:
    rng = np.random.default_rng(0)
    gray = rng.integers(0, 255, (height, 64), dtype=np.uint8)
    tiles = plan_tiles(gray, tile_height=2048, overlap=192)
    assert tiles[0].y0 == 0 and tiles[-1].y1 == height
    for a, b in itertools.pairwise(tiles):
        assert b.y0 == a.y1 - 192  # exact overlap
        assert b.y0 > a.y0  # progress
    assert all(t.height <= 2048 for t in tiles)


def test_cuts_prefer_gutters() -> None:
    gray = np.random.default_rng(1).integers(0, 255, (5000, 100), dtype=np.uint8)
    gray[1850:1950] = 255  # a blank gutter inside the search window
    tiles = plan_tiles(gray, tile_height=2048, overlap=192)
    assert 1850 <= tiles[0].y1 <= 1950


def test_merge_prefers_complete_copy() -> None:
    page_h = 4000
    t1, t2 = Tile(0, 2000), Tile(1808, 4000)
    cut = BBox(10, 1900, 110, 2000)  # touches t1's interior bottom edge
    full = BBox(10, 1900, 110, 2100)  # complete inside t2
    items = [(t1, ("a", cut)), (t2, ("b", full)), (t1, ("c", BBox(0, 10, 50, 60)))]
    merged = merge_tiled(items, box_of=lambda it: it[1], page_height=page_h)
    names = sorted(m[0][0] for m in merged)
    assert names == ["b", "c"]
    assert all(complete for _, complete in merged)
    assert not is_complete_in_tile(cut, t1, page_h) and is_complete_in_tile(full, t2, page_h)


def test_merge_flags_blocks_cut_everywhere() -> None:
    t1, t2 = Tile(0, 1000), Tile(900, 2000)
    a = BBox(0, 850, 100, 1000)
    b = BBox(0, 900, 100, 1150)
    merged = merge_tiled([(t1, a), (t2, b)], box_of=lambda x: x, page_height=2000)
    # a touches t1's bottom cut and b starts on t2's top cut: cut in every tile.
    assert len(merged) == 1 and merged[0] == (b, False)


def test_webtoon_seam_ground_truth_once(cjk_ready: Path, cache_dir: Path) -> None:
    """Simulated per-tile detector (GT boxes clipped to each tile) → merged exactly once."""
    from manga_ar.synth import webtoon_strip

    page = webtoon_strip(0, cache_dir)
    gray = cv2.cvtColor(page.image, cv2.COLOR_RGB2GRAY)
    tiles = plan_tiles(gray, 2048, 192)
    items: list[tuple[Tile, BBox]] = []
    for tile in tiles:
        for r in page.regions:
            box = r.text_bbox
            inter = box.intersection(BBox(0, tile.y0, page.width, tile.y1))
            if inter is not None:
                items.append((tile, inter))
    merged = merge_tiled(items, box_of=lambda b: b, page_height=page.height)
    assert len(merged) == len(page.regions)
    for r in page.regions:
        hits = [m for m, _ in merged if m.iou(r.text_bbox) > 0.9]
        assert len(hits) == 1, r.text
