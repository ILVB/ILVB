"""Translation metrics: sacreBLEU, chrF++, METEOR (Arabic adaptation), paired bootstrap."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, cast

from sacrebleu.metrics.bleu import BLEU
from sacrebleu.metrics.chrf import CHRF

TOKENIZE = "intl"  # Unicode-aware punctuation splitting (Arabic ، ؟ etc.)
_WORD = re.compile(r"\w+", re.UNICODE)


def _streams(refs: Sequence[Sequence[str]]) -> list[list[str]]:
    """Per-segment reference lists → sacrebleu reference streams (pad by repetition)."""
    n = max(len(r) for r in refs)
    return [[r[min(k, len(r) - 1)] for r in refs] for k in range(n)]


def bleu(hyps: Sequence[str], refs: Sequence[Sequence[str]]) -> float:
    return float(BLEU(tokenize=TOKENIZE).corpus_score(list(hyps), _streams(refs)).score)


def chrf_pp(hyps: Sequence[str], refs: Sequence[Sequence[str]]) -> float:
    return float(CHRF(word_order=2).corpus_score(list(hyps), _streams(refs)).score)


class _NoSynonyms:
    """METEOR's synonym stage is unavailable for Arabic (no Arabic WordNet here)."""

    def synsets(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        return []


def meteor(hyps: Sequence[str], refs: Sequence[Sequence[str]]) -> float:
    """Mean segment METEOR with exact + ISRI-stem matching (Arabic adaptation)."""
    from nltk.stem.isri import ISRIStemmer
    from nltk.translate.meteor_score import meteor_score

    stemmer = ISRIStemmer()
    scores = []
    for hyp, rs in zip(hyps, refs, strict=True):
        h = _WORD.findall(hyp)
        rr = [_WORD.findall(r) for r in rs]
        scores.append(meteor_score(rr, h, stemmer=stemmer, wordnet=_NoSynonyms()) if h else 0.0)
    return float(sum(scores) / len(scores)) if scores else 0.0


def paired_bootstrap(
    baseline: Sequence[str], candidate: Sequence[str], refs: Sequence[Sequence[str]],
    n_samples: int = 1000,
) -> dict[str, dict[str, float]]:  # fmt: skip
    """Paired bootstrap (sacrebleu) for BLEU and chrF++: scores and candidate p-values."""
    from sacrebleu.significance import PairedTest

    test = PairedTest(
        [("baseline", list(baseline)), ("candidate", list(candidate))],
        {"BLEU": BLEU(tokenize=TOKENIZE), "chrF++": CHRF(word_order=2)},
        _streams(refs), test_type="bs", n_samples=n_samples,
    )  # fmt: skip
    _sigs, scores = test()
    out: dict[str, dict[str, float]] = {}
    for key, name in (("BLEU", "bleu"), ("chrF2++", "chrf_pp")):
        base, cand = (cast(Any, r) for r in scores[key])
        out[name] = {"baseline": float(base.score), "candidate": float(cand.score),
                     "p_value": float(cand.p_value)}  # fmt: skip
    return out
