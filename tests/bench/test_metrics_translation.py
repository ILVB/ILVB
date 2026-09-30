"""Translation metrics against hand-computed values."""

from __future__ import annotations

import math

import pytest

from benchmarks.metrics import translation as tr


def test_reference_streams_pad_by_repetition() -> None:
    assert tr._streams([["a", "b"], ["c"]]) == [["a", "c"], ["b", "c"]]


def test_bleu_brevity_penalty_hand_value() -> None:
    # All n-gram precisions are 1; BP = exp(1 - 6/4).
    score = tr.bleu(["a b c d"], [["a b c d e f"]])
    assert score == pytest.approx(100 * math.exp(1 - 6 / 4), abs=1e-6)
    assert tr.bleu(["مرحبا يا صديقي العزيز"], [["مرحبا يا صديقي العزيز"]]) == pytest.approx(100)


def test_bleu_uses_best_of_multiple_references() -> None:
    hyp = ["أين أنت الآن يا صديقي"]
    single = tr.bleu(hyp, [["أين كنت أمس"]])
    multi = tr.bleu(hyp, [["أين كنت أمس", "أين أنت الآن يا صديقي"]])
    assert multi == pytest.approx(100) and single < multi


def test_chrf_pp_bounds() -> None:
    assert tr.chrf_pp(["قطة سوداء"], [["قطة سوداء"]]) == pytest.approx(100)
    assert tr.chrf_pp(["xyz"], [["قطة"]]) == pytest.approx(0)


def test_meteor_exact_and_stem_matching() -> None:
    # 3 exact matches in one chunk: Fmean = 1, penalty = 0.5 * (1/3)**3.
    assert tr.meteor(["ذهب الولد مسرعا"], [["ذهب الولد مسرعا"]]) == pytest.approx(
        1 - 0.5 * (1 / 3) ** 3, abs=1e-4
    )
    stem_match = tr.meteor(["الكتاب جميل"], [["كتاب جميل"]])
    assert stem_match > tr.meteor(["قلم جميل"], [["كتاب جميل"]])  # ISRI stem match counts
    assert tr.meteor([""], [["كتاب"]]) == 0.0


def test_paired_bootstrap_significance() -> None:
    refs = [[f"هذه هي الجملة رقم {i} في الاختبار"] for i in range(30)]
    cand = [r[0] for r in refs]
    base = [f"كلام آخر مختلف {i}" for i in range(30)]
    out = tr.paired_bootstrap(base, cand, refs, n_samples=200)
    assert out["bleu"]["candidate"] == pytest.approx(100)
    assert out["bleu"]["candidate"] > out["bleu"]["baseline"]
    assert out["bleu"]["p_value"] < 0.05 and out["chrf_pp"]["p_value"] < 0.05
