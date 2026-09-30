"""0.3.2: synthetic generator determinism and ground-truth consistency."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks import rle
from benchmarks.generators.draw import NO_LINE_START, columns, dilate, wrap
from benchmarks.generators.fonts import FontError, font_path
from benchmarks.generators.page import CATEGORIES, H, W, make_page
from benchmarks.generators.texts import TextPools, bank, split_of_keys
from benchmarks.schema import GtPage


@pytest.fixture(scope="module")
def pools() -> TextPools:
    try:
        font_path("noto_jp")
    except FontError as exc:
        pytest.skip(f"generator fonts unavailable: {exc}")
    return TextPools("dev")


def _page(category: str, pools: TextPools, seed: int = 11) -> tuple:
    lang = "ja" if category in ("vertical_ja", "flat_white") else "zh"
    return make_page(f"t-{category}", category, lang, "dev", seed, pools)


def test_deterministic(pools: TextPools) -> None:
    a_img, a_clean, a_gt = _page("tails_overlap", pools)
    b_img, b_clean, b_gt = _page("tails_overlap", pools)
    assert np.array_equal(a_img, b_img) and np.array_equal(a_clean, b_clean) and a_gt == b_gt


@pytest.mark.parametrize("category", CATEGORIES)
def test_ground_truth_is_consistent(category: str, pools: TextPools) -> None:
    img, clean, gt = _page(category, pools)
    page = GtPage.model_validate({**gt, "image": "x.png", "clean": "y.png"})
    assert len(page.regions) >= 3
    assert [r.reading_order for r in page.regions] == list(range(1, len(page.regions) + 1))
    union = np.zeros((H, W), bool)
    for r in page.regions:
        text = rle.decode(r.text_mask, (H, W))
        safe = rle.decode(r.safe_mask, (H, W))
        assert text.any() and safe.any() and r.text
        union |= text
        if r.bubble_mask is not None:
            bubble = rle.decode(r.bubble_mask, (H, W))
            assert (text & ~bubble).sum() == 0, "lettering must sit inside its bubble"
            assert (safe & ~bubble).sum() == 0, "the safe area must sit inside the bubble"
    # outside the lettering the page is exactly the clean background
    outside = ~dilate(union, 1)
    assert np.array_equal(img[outside], clean[outside])
    assert not np.array_equal(img, clean)


def test_kinsoku_and_wrapping(pools: TextPools) -> None:
    cols = columns("大丈夫、心配しないで。", 4)
    assert (
        all(c[0] not in NO_LINE_START for c in cols) and "".join(cols) == "大丈夫、心配しないで。"
    )
    for text in ("今天好像要下雨。今天好像要下雨。", "谢谢你，帮了大忙。"):
        lines = wrap(text, "noto_sc", 28, 80, "zh")
        assert all(ln[0] not in NO_LINE_START for ln in lines) and "".join(lines) == text


def test_splits_disjoint_and_balanced() -> None:
    ids = list(bank()["phrases"])
    assignment = split_of_keys(ids)
    sets = {s: {k for k, v in assignment.items() if v == s} for s in ("dev", "val", "test")}
    assert all(len(v) == 32 for v in sets.values())
    assert not (sets["dev"] & sets["val"]) and not (sets["val"] & sets["test"])
    assert not (sets["dev"] & sets["test"])


def test_dialogue_regions_carry_silver_references(pools: TextPools) -> None:
    from benchmarks.generators.texts import silver_refs

    _img, _clean, gt = make_page("t-refs", "flat_white", "zh", "dev", 5, pools, silver_refs())
    for r in gt["regions"]:
        assert r["reference_kind"] == "silver" and r["references_ar"], r["region_id"]
    refs = silver_refs()
    assert refs["m41"]["zh"][0].startswith("شياو") and refs["m41"]["all"][0].startswith("هانا")
