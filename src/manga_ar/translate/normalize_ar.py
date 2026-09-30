"""Normalisation of engine output before typesetting (S5 ``normalize_ar``, E20).

Removes bidi controls (display spoofing / double processing), zero-width characters and
stray tatweel; unifies Persian/Urdu letter variants; maps Latin/CJK punctuation to
Arabic; applies the digit policy; re-appends preserved decorations; caps runaway length.
"""

from __future__ import annotations

import re
import unicodedata

BIDI_CONTROLS = "‎‏‪‫‬‭‮⁦⁧⁨⁩؜"
ZERO_WIDTH = "​‌‍⁠﻿"
TATWEEL = "ـ"
LETTER_VARIANTS = {
    "ک": "ك",  # KEHEH → KAF
    "ی": "ي",  # FARSI YEH → YEH
    "ہ": "ه",  # HEH GOAL → HEH
    "ھ": "ه",  # HEH DOACHASHMEE → HEH
    "گ": "ك",  # GAF → KAF (not an Arabic-language letter)
}
QUOTES = {"「": "«", "」": "»", "『": "«", "』": "»", "“": "«", "”": "»"}
WESTERN = "0123456789"
INDIC = "٠١٢٣٤٥٦٧٨٩"
EXT_INDIC = "۰۱۲۳۴۵۶۷۸۹"
_ARABIC = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
_STRIP = str.maketrans("", "", BIDI_CONTROLS + ZERO_WIDTH + TATWEEL)


def is_arabic_char(ch: str) -> bool:
    return bool(_ARABIC.match(ch)) and unicodedata.category(ch).startswith("L")


def arabic_ratio(text: str) -> float:
    """Share of letters that are Arabic."""
    letters = [c for c in text if unicodedata.category(c).startswith("L")]
    if not letters:
        return 0.0
    return sum(1 for c in letters if is_arabic_char(c)) / len(letters)


def has_cjk(text: str) -> bool:
    return any(
        0x3040 <= ord(c) <= 0x30FF or 0x3400 <= ord(c) <= 0x9FFF or 0xAC00 <= ord(c) <= 0xD7A3
        for c in text
    )


def strip_controls(text: str) -> str:
    return text.translate(_STRIP)


def normalize_ar(
    text: str,
    digits: str = "western",
    decorations: list[str] | None = None,
    max_chars: int = 400,
) -> str:
    t = unicodedata.normalize("NFKC", text)  # also un-shapes presentation forms
    t = strip_controls(t)
    t = "".join(LETTER_VARIANTS.get(c, c) for c in t)
    t = "".join(QUOTES.get(c, c) for c in t)
    t = re.sub(r"(?:\.\s*){3,}|。{2,}|…+|・{3,}", "…", t)
    t = re.sub(r"(?<!\d),|,(?!\d)", "،", t)  # keep decimal commas (3,5)
    t = t.replace(";", "؛").replace("?", "؟")
    t = re.sub(r"\s+([،؛؟!.:…»])", r"\1", t)  # no space before punctuation
    t = re.sub(r"([«])\s+", r"\1", t)
    t = re.sub(r"([،؛؟!:])(?=[^\s\d،؛؟!.…»:])", r"\1 ", t)
    if digits == "western":
        t = t.translate(str.maketrans(INDIC + EXT_INDIC, WESTERN * 2))
    elif digits == "arabic_indic":
        t = t.translate(str.maketrans(WESTERN + EXT_INDIC, INDIC * 2))
    else:
        raise ValueError(f"unknown digit policy {digits!r}")
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > max_chars:
        cut = t.rfind(" ", 0, max_chars)
        t = t[: cut if cut > max_chars // 2 else max_chars].rstrip() + "…"
    if decorations:
        t = f"{t} {''.join(decorations)}".strip()
    return t
