from __future__ import annotations

from pathlib import Path

import numpy as np

from manga_ar import synth
from manga_ar.config import load_config
from manga_ar.detect.bubble import FloodBubbleSegmenter
from manga_ar.detect.classical import ClassicalDetector
from manga_ar.detect.reading_order import assign_reading_order, default_mode, find_panels
from manga_ar.metrics import match_boxes
from manga_ar.schemas import BBox, Region, RegionType

CFG = load_config(environ={})


def test_reading_order_gate(cjk_ready: Path, cache_dir: Path) -> None:
    pages = [
        synth.basic_page(lang, s, cache_dir) for lang in ("ja", "ko", "zh") for s in (40, 41, 42)
    ]
    pages += [
        synth.variety_page("ja", 40, cache_dir),
        synth.demo_page(cache_dir),
        synth.basic_page("ja", 43, cache_dir, vertical=False),
    ]
    exact = 0
    det, seg = ClassicalDetector(CFG.detect), FloodBubbleSegmenter(CFG.detect)
    for page in pages:
        regions = [
            r
            for r in seg.segment(page.image, det.detect(page.image), page.name)
            if r.type != RegionType.SFX
        ]
        regions = assign_reading_order(regions, page.image, page.reading_order)
        m = match_boxes([g.text_bbox for g in page.regions], [r.bbox for r in regions], 0.5)
        order = [gi for gi, rj in sorted(m, key=lambda t: regions[t[1]].reading_order)]
        exact += order == sorted(order) and len(m) == len(page.regions)
    assert exact / len(pages) >= 0.95, (exact, len(pages))


def test_find_panels_grid_order() -> None:
    page = np.full((400, 400), 255, np.uint8)
    for x0, y0 in ((10, 10), (210, 10), (10, 210), (210, 210)):
        page[y0 : y0 + 180, x0 : x0 + 180] = 0
        page[y0 + 3 : y0 + 177, x0 + 3 : x0 + 177] = 200
    rtl = find_panels(page, "manga_rtl")
    ltr = find_panels(page, "comic_ltr")
    assert len(rtl) == 4 and rtl[0].x0 > rtl[1].x0 and rtl[0].y0 < rtl[2].y0
    assert ltr[0].x0 < ltr[1].x0


def test_webtoon_top_to_bottom() -> None:
    rgb = np.full((1000, 300, 3), 255, np.uint8)
    regions = [
        Region(
            id=f"r{i}",
            type=RegionType.BUBBLE,
            bbox=BBox(10 + 50 * (i % 2), y, 60 + 50 * (i % 2), y + 30),
        )
        for i, y in enumerate((700, 100, 400))
    ]
    ordered = assign_reading_order(regions, rgb, "webtoon_ttb")
    assert [r.bbox.y0 for r in ordered] == [100, 400, 700]
    assert default_mode("ja") == "manga_rtl" and default_mode("ko") == "webtoon_ttb"
    assert default_mode(None) == "manga_rtl"
