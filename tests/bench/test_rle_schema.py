"""Benchmark RLE and schema contracts."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from benchmarks import rle
from benchmarks.schema import GtRegion, PageResult


@settings(max_examples=40, deadline=None)
@given(st.integers(1, 30), st.integers(1, 30), st.integers(0, 10_000))
def test_rle_round_trip(h: int, w: int, seed: int) -> None:
    mask = np.random.default_rng(seed).random((h, w)) > 0.6
    back = rle.decode(rle.encode(mask), (h, w))
    assert np.array_equal(back, mask)


def test_rle_empty_and_corrupt() -> None:
    assert rle.encode(np.zeros((4, 4), bool)) is None
    assert not rle.decode(None, (3, 3)).any()
    with pytest.raises(ValueError, match="covers"):
        rle.decode({"bbox": [0, 0, 2, 2], "runs": [1, 1]}, (4, 4))


def test_schemas_are_strict() -> None:
    mask = rle.encode(np.ones((2, 2), bool))
    region = GtRegion(
        region_id="p-r1", category="flat", type="dialogue", lang="ja", text="あ",
        text_polygon=[(0, 0), (2, 0), (2, 2), (0, 2)], text_mask=mask, safe_mask=mask,
        reading_order=1,
    )  # fmt: skip
    assert region.background == "flat"
    with pytest.raises(ValueError):
        PageResult(page_id="p", version="baseline", mode="erase", bogus=1)  # type: ignore[call-arg]
