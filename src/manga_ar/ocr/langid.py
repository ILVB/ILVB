"""Source-language identification by Unicode-script voting (S3, E6)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable


def script_of(ch: str) -> str | None:
    cp = ord(ch)
    if 0x3040 <= cp <= 0x30FF or 0x31F0 <= cp <= 0x31FF or 0xFF66 <= cp <= 0xFF9D:
        return "kana"
    if 0xAC00 <= cp <= 0xD7A3 or 0x1100 <= cp <= 0x11FF or 0x3130 <= cp <= 0x318F:
        return "hangul"
    if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0xF900 <= cp <= 0xFAFF:
        return "han"
    if ch.isascii() and ch.isalpha():
        return "latin"
    if ch.isdigit():
        return "digit"
    return None


def script_counts(text: str) -> Counter[str]:
    return Counter(s for s in (script_of(c) for c in text) if s)


def classify_text(text: str) -> str | None:
    """Hiragana/Katakana ⇒ ja; Hangul ⇒ ko; Han-only ⇒ zh; else None."""
    counts = script_counts(text)
    if counts["hangul"] and counts["hangul"] >= counts["kana"]:
        return "ko"
    if counts["kana"]:
        return "ja"
    if counts["han"]:
        return "zh"
    return None


def vote_language(texts: Iterable[str], weights: Iterable[float] | None = None) -> str | None:
    """Document-level vote: each text contributes its classification × weight."""
    tally: dict[str, float] = {}
    ws = list(weights) if weights is not None else None
    for k, text in enumerate(texts):
        lang = classify_text(text)
        if lang is not None:
            tally[lang] = tally.get(lang, 0.0) + (ws[k] if ws is not None else 1.0)
    if not tally:
        return None
    return max(tally, key=lambda key: tally[key])


def is_passthrough(text: str) -> bool:
    """Latin-only / digits-only / punctuation-only text (left untouched by default)."""
    counts = script_counts(text)
    return not (counts["kana"] or counts["hangul"] or counts["han"])


def script_consistency(text: str, lang: str) -> float:
    """Share of letters belonging to scripts expected for ``lang``."""
    counts = script_counts(text)
    letters = sum(v for k, v in counts.items() if k not in {"digit"})
    if letters == 0:
        return 0.0
    expected = {"ja": {"kana", "han"}, "zh": {"han"}, "ko": {"hangul"}, "en": {"latin"}}[lang]
    return sum(counts[s] for s in expected) / letters
