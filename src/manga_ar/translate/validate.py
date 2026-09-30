"""Suspicious-translation detection (S5, E3)."""

from __future__ import annotations

import unicodedata

from manga_ar.config import TranslateConfig
from manga_ar.translate.normalize_ar import arabic_ratio, has_cjk


def _norm(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).split())


def translation_problems(source: str, output: str, cfg: TranslateConfig) -> list[str]:
    """Reasons why ``output`` is not an acceptable Arabic rendering of ``source``."""
    out = output.strip()
    if not out:
        return ["empty"]
    reasons = []
    if _norm(out) == _norm(source):
        reasons.append("identical-to-source")
    ratio = arabic_ratio(out)
    if ratio < cfg.min_arabic_ratio:
        reasons.append(f"arabic-ratio({ratio:.2f})")
    if has_cjk(out):
        reasons.append("cjk-remaining")
    if len(source) >= 2 and len(out) > cfg.max_length_ratio * len(source) + 20:
        reasons.append(f"length-ratio({len(out)}/{len(source)})")
    return reasons
