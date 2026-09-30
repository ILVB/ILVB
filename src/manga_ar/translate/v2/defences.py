"""Output defences applied to every v2 translation (E-12, E-13).

On top of v0.1.0's checks (script ratio, identical output, leftover CJK, length), a
rendering is rejected for refusal or safety boilerplate, meta-commentary that cannot be
stripped, repetition loops, and instruction echo (Latin-script words that are not in the
source). A rejected rendering is never typeset; the router falls back instead.
"""

from __future__ import annotations

import re

from manga_ar.config import TranslateConfig
from manga_ar.translate.validate import translation_problems

_REFUSAL = re.compile(
    r"\b(i'?m sorry|i am sorry|i can(?:no|')t|i am unable|as an ai|language model|"
    r"i (?:won't|will not) (?:translate|help))\b"
    r"|عذرًا، لا|عذرا، لا|لا أستطيع|لا يمكنني|كنموذج لغوي|بصفتي نموذج|كذكاء اصطناعي",
    re.IGNORECASE,
)
_META = re.compile(
    r"^\s*(?:here(?:'s| is) (?:the |your )?translation|translation|the translation is|"
    r"الترجمة|ترجمة)\s*[:：\-–]\s*",
    re.IGNORECASE,
)
_LOOP = re.compile(r"(\S{2,}(?:\s+\S+){0,2}?)(?:\s+\1){3,}")
_LATIN_WORD = re.compile(r"[A-Za-z]{3,}")


def strip_meta(text: str) -> str:
    """Remove a leading 'Here is the translation:' style label and wrapping quotes."""
    out = _META.sub("", text.strip(), count=1).strip()
    if len(out) >= 2 and out[0] + out[-1] in {'""', "«»", "“”", "''"}:
        out = out[1:-1].strip()
    return out


def output_problems(source: str, output: str, cfg: TranslateConfig) -> list[str]:
    reasons = translation_problems(source, output, cfg)
    if _REFUSAL.search(output):
        reasons.append("refusal-boilerplate")
    if _META.match(output):
        reasons.append("meta-commentary")
    if _LOOP.search(output) and not _LOOP.search(source):
        reasons.append("repetition-loop")
    echoed = {w.lower() for w in _LATIN_WORD.findall(output)}
    echoed -= {w.lower() for w in _LATIN_WORD.findall(source)}
    if len(echoed) >= 2:
        reasons.append(f"instruction-echo({','.join(sorted(echoed)[:3])})")
    return reasons


def accept(source: str, raw: str, cfg: TranslateConfig) -> tuple[str | None, list[str]]:
    """(rendering, []) when acceptable after stripping meta labels, else (None, reasons)."""
    candidate = strip_meta(raw)
    reasons = output_problems(source, candidate, cfg)
    return (candidate, []) if not reasons else (None, reasons)
