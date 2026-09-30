"""normalize_ar golden tests (ARVS L1 for normalisation, E20)."""

from __future__ import annotations

import pytest

from manga_ar.translate.normalize_ar import BIDI_CONTROLS, arabic_ratio, has_cjk, normalize_ar


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("مرحبا,كيف حالك?", "مرحبا، كيف حالك؟"),
        ("أولاً; ثانياً", "أولاً؛ ثانياً"),
        ("انتظر ...", "انتظر…"),
        ("انتظر。。。", "انتظر…"),
        ("「مرحبا」", "«مرحبا»"),
        ("كتاب ک ی", "كتاب ك ي"),  # Persian keheh/yeh → Arabic kaf/yeh
        ("جمـــيل", "جميل"),  # tatweel removed
        ("‮مرحبا‬", "مرحبا"),  # RLO/PDF spoofing removed
        ("a\u200f\u0628\u2066\u062a\u2069", "a\u0628\u062a"),  # bidi controls stripped
        ("​نص‍", "نص"),
        ("لا   تذهب  !", "لا تذهب!"),
        ("السعر 3,5 ريال", "السعر 3,5 ريال"),  # decimal comma kept
        ("الساعة ٥", "الساعة 5"),
        ("الساعة ۵", "الساعة 5"),
        ("ﻣﺮﺣﺒﺎ", "مرحبا"),  # presentation forms un-shaped (no double shaping later)
    ],
)
def test_golden(raw: str, expected: str) -> None:
    assert normalize_ar(raw) == expected


def test_digit_policy_and_decorations() -> None:
    assert normalize_ar("الساعة 10", digits="arabic_indic") == "الساعة ١٠"
    assert normalize_ar("شكرا", decorations=["♡", "♪"]) == "شكرا ♡♪"
    with pytest.raises(ValueError):
        normalize_ar("x", digits="roman")


def test_no_bidi_controls_survive() -> None:
    dirty = "".join(f"ب{c}" for c in BIDI_CONTROLS)
    assert not any(c in normalize_ar(dirty) for c in BIDI_CONTROLS)


def test_length_cap() -> None:
    long = " ".join(["كلمة"] * 200)
    out = normalize_ar(long, max_chars=100)
    assert len(out) <= 101 and out.endswith("…")


def test_ratios() -> None:
    assert arabic_ratio("مرحبا بالعالم") == 1.0
    assert arabic_ratio("hello") == 0.0
    assert 0.4 < arabic_ratio("مرحبا OK") < 0.8
    assert has_cjk("abc 日本") and not has_cjk("مرحبا")
