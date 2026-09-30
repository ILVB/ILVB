"""Text and detection metrics against hand-computed values."""

from __future__ import annotations

import pytest

from benchmarks.metrics.detection import DetectionCounts, box_of, iou, match, order_hits
from benchmarks.metrics.text import EditCounts, char_edits, normalize, word_edits


def test_normalize_nfkc_and_whitespace() -> None:
    assert normalize("ＡＢ　ｃ\n１") == "ABc1"  # full-width → ASCII, whitespace dropped


def test_char_edits_micro_average() -> None:
    a = char_edits("你真的要去吗？", "你真要去吗?")  # 的 deleted; ？ vs ? equal after NFKC
    assert a == EditCounts(1, 7)
    b = char_edits("等一下", "")
    assert b == EditCounts(3, 3)
    total = a + b
    assert total.rate == pytest.approx(4 / 10)  # micro-average, not mean of rates
    assert char_edits("", "x") == EditCounts(1, 0)


def test_word_edits() -> None:
    assert word_edits("Are you really going?", "Are you going?") == EditCounts(1, 4)
    assert word_edits("Wait", "") == EditCounts(1, 1)


def test_iou_and_matching() -> None:
    a, b = (0, 0, 10, 10), (5, 0, 15, 10)
    assert iou(a, b) == pytest.approx(50 / 150)
    assert box_of([(3, 4), (9, 1), (5, 8)]) == (3, 1, 9, 8)
    gt = [(0, 0, 10, 10), (20, 20, 30, 30)]
    pred = [(0, 0, 10, 9), (1, 1, 10, 10), (50, 50, 60, 60)]
    pairs = match(gt, pred)
    assert [(g, p) for g, p, _ in pairs] == [(0, 0)]  # one-to-one; (0,1) loses; far box FP
    counts = DetectionCounts(len(pairs), len(gt), len(pred))
    assert counts.precision == pytest.approx(1 / 3) and counts.recall == pytest.approx(1 / 2)
    assert counts.f1 == pytest.approx(0.4)


def test_order_hits_uses_ranks_among_matched() -> None:
    assert order_hits([(0, 5), (1, 7), (2, 9)]) == (3, 3)  # gaps in predicted order are fine
    assert order_hits([(0, 1), (1, 0), (2, 2)]) == (1, 3)  # first two swapped
    assert order_hits([]) == (0, 0)
