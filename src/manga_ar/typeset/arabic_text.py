"""Arabic text processing for the BASIC render path (A1, A3, A12).

Pipeline per *line* (never per paragraph): logical text → ``reshape`` (contextual forms,
lam-alef ligatures) → bidi visual order with base direction forced RTL → draw LTR.

python-bidi 0.6's Rust ``get_display`` resolves levels correctly (including bracket
pairs, N0) but does not apply rule L4 (mirroring), so ``على (OK)`` would show ``)OK(``.
The shim obtains the resolved embedding levels and performs L2 (reordering) and L4
(mirroring) itself. Older pure-Python python-bidi releases mirror on their own and are
used as-is (fallback).
"""

from __future__ import annotations

import functools
import re
import unicodedata

import arabic_reshaper

from manga_ar.logging_setup import get_logger

log = get_logger(__name__)

HARAKAT = re.compile("[ً-ٰٟۖ-ۭ]")
_LEVEL = re.compile(r"Level\(\s*(\d+),?\s*\)")

# Explicit reshaper configuration: only the four lam-alef letter ligatures; word and
# sentence ligatures (e.g. ALLAH → U+FDF2) are off so joining stays predictable and
# fonts need nothing exotic.
_BASE_CONFIG: dict[str, object] = {
    "language": "Arabic",
    "support_ligatures": True,
    "support_zwj": True,
    "delete_tatweel": True,
    "shift_harakat_position": False,
    "ARABIC LIGATURE ALLAH": False,
    "ARABIC LIGATURE LAM WITH ALEF": True,
    "ARABIC LIGATURE LAM WITH ALEF WITH HAMZA ABOVE": True,
    "ARABIC LIGATURE LAM WITH ALEF WITH HAMZA BELOW": True,
    "ARABIC LIGATURE LAM WITH ALEF WITH MADDA ABOVE": True,
}


@functools.lru_cache(maxsize=16)
def reshaper_for(
    unshaped_isolated: bool, strip_harakat: bool = True
) -> arabic_reshaper.ArabicReshaper:
    """Reshaper per font capability: fonts lacking isolated presentation forms get base
    letters instead (they render as isolated glyphs in every font)."""
    config = dict(_BASE_CONFIG)
    config["delete_harakat"] = strip_harakat
    config["use_unshaped_instead_of_isolated"] = unshaped_isolated
    return arabic_reshaper.ArabicReshaper(configuration=config)


def strip_harakat(text: str) -> str:
    return HARAKAT.sub("", text)


# ----------------------------------------------------------------------------- bidi
def _mirror_char(ch: str) -> str:
    try:
        from bidi.mirror import MIRRORED

        return str(MIRRORED.get(ch, ch))
    except ImportError:  # pragma: no cover - very old python-bidi
        return {
            "(": ")",
            ")": "(",
            "[": "]",
            "]": "[",
            "{": "}",
            "}": "{",
            "<": ">",
            ">": "<",
            "«": "»",
            "»": "«",
        }.get(ch, ch)


def _rust_levels(text: str) -> list[int] | None:
    """Resolved embedding level per character from python-bidi's Rust engine."""
    try:
        from bidi.bidi import get_display_inner
    except ImportError:
        return None
    debug = get_display_inner(text, "R", True)
    if not isinstance(debug, str) or "levels:" not in debug:
        return None
    start = debug.index("levels:")
    end = debug.find("paragraphs:", start)
    block = debug[start : end if end > 0 else len(debug)]
    per_byte = [int(v) for v in _LEVEL.findall(block)]
    encoded = [len(ch.encode("utf-8")) for ch in text]
    if len(per_byte) != sum(encoded):
        return None
    levels, pos = [], 0
    for n in encoded:
        levels.append(per_byte[pos])
        pos += n
    return levels


def reorder(text: str, levels: list[int], mirror: bool = True) -> str:
    """UBA L2 (reverse runs from the highest level down to the lowest odd level) + L4."""
    chars = list(text)
    if mirror:
        chars = [_mirror_char(c) if lv % 2 == 1 else c for c, lv in zip(chars, levels, strict=True)]
    if not levels:
        return ""
    order = list(range(len(chars)))
    highest = max(levels)
    lowest_odd = min((lv for lv in levels if lv % 2 == 1), default=highest + 1)
    for level in range(highest, lowest_odd - 1, -1):
        i = 0
        while i < len(order):
            if levels[order[i]] >= level:
                j = i
                while j < len(order) and levels[order[j]] >= level:
                    j += 1
                order[i:j] = order[i:j][::-1]
                i = j
            else:
                i += 1
    return "".join(chars[k] for k in order)


@functools.lru_cache(maxsize=1)
def _mirrorable() -> frozenset[str]:
    try:
        from bidi.mirror import MIRRORED

        return frozenset(MIRRORED)
    except ImportError:  # pragma: no cover - very old python-bidi
        return frozenset("()[]{}<>«»")


def visual_order(text: str) -> str:
    """Visual (left-to-right drawing) order of one logical line, base direction RTL."""
    if not text:
        return text
    if not any(c in _mirrorable() for c in text):
        # Nothing to mirror: the Rust reordering is already complete (and fast).
        from bidi import get_display

        return str(get_display(text, base_dir="R"))
    levels = _rust_levels(text)
    if levels is not None:
        return reorder(text, levels)
    from bidi.algorithm import get_display  # pure-Python releases mirror themselves

    return str(get_display(text, base_dir="R"))


@functools.lru_cache(maxsize=16384)
def shape_line(logical: str, unshaped_isolated: bool = False, harakat: bool = True) -> str:
    """Logical line → shaped visual string ready for BASIC-layout drawing (pure, cached:
    the fit search measures the same candidate lines many times, A13)."""
    shaped = reshaper_for(unshaped_isolated, harakat).reshape(logical)
    return visual_order(shaped)


def form_of(ch: str) -> str:
    """ "isolated" | "initial" | "medial" | "final" | "base" | "other" (for tests/debug)."""
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return "other"
    for form in ("ISOLATED", "INITIAL", "MEDIAL", "FINAL"):
        if name.endswith(f"{form} FORM"):
            return form.lower()
    return "base" if name.startswith("ARABIC LETTER") else "other"
