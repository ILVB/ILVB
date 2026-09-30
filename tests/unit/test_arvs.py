"""Arabic Rendering Verification Suite (T4): proves text is neither backwards nor
disconnected. Negative controls demonstrate that each check can fail."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, features

from manga_ar.typeset.arabic_text import form_of, reshaper_for, shape_line, visual_order
from manga_ar.typeset.fonts import FontInfo, FontRegistry, load_font
from manga_ar.typeset.textline import FontChoice, RaqmChoice

CORPUS = json.loads(
    (Path(__file__).parents[1] / "data" / "arabic_corpus.json").read_text(encoding="utf-8")
)
CANONICAL = [c["text"] for c in CORPUS["canonical"]]
ALL_TEXTS = CANONICAL + CORPUS["extra"]
REG = FontRegistry()
BASIC_FONTS = [
    f for f in REG.fonts.values() if f.role == "arabic" and f.basic_ok and f.source == "vendored"
]
SYMBOLS = REG.symbol_chain(["NotoSansSymbols2", "NotoSansSymbols"])

# ---------------------------------------------------------------- L2 shaping oracle
# Independent joining-type table (Unicode ArabicShaping), hand-coded for review:
# R = joins only to the preceding letter; U = non-joining; everything else is dual (D).
RIGHT_JOINING = set("اأإآؤةدذرزو")
NON_JOINING = set("ء")
ALEFS = set("اأإآ")
LIGATURE_NAMES = {
    "ا": "LAM WITH ALEF",
    "أ": "LAM WITH ALEF WITH HAMZA ABOVE",
    "إ": "LAM WITH ALEF WITH HAMZA BELOW",
    "آ": "LAM WITH ALEF WITH MADDA ABOVE",
}
L2_WORDS = [
    "مرحبا",
    "بالعالم",
    "لا",
    "تذهب",
    "هل",
    "أنت",
    "بخير",
    "الحمد",
    "لله",
    "محمد",
    "بلا",
    "شك",
    "ولا",
    "تردد",
    "سأنتظرك",
    "عند",
    "الساعة",
    "مساء",
    "اضغط",
    "على",
    "ثم",
    "انتظر",
    "ثوان",
    "كلا",
    "يمكنني",
    "ذلك",
    "للهول",
    "ماذا",
    "حدث",
    "أسامحك",
    "فعلته",
    "وبأصدقائي",
    "وسأجعلك",
    "تدفع",
    "الثمن",
    "غاليا",
    "مهما",
    "طال",
    "الزمن",
    "الأمل",
    "الإسلام",
    "الآن",
    "بلأ",
    "بلإ",
    "بلآ",
    "سلام",
    "لؤلؤة",
    "شيء",
    "مسؤول",
    "ضوء",
]


def joining(ch: str) -> str:
    if ch in NON_JOINING:
        return "U"
    return "R" if ch in RIGHT_JOINING else "D"


def expected_forms(word: str) -> list[tuple[str, str]]:
    """(letter-name, form) for each output glyph, lam-alef ligatures included."""
    out: list[tuple[str, str]] = []
    i = 0
    while i < len(word):
        ch = word[i]
        prev_dual = (
            i > 0
            and joining(word[i - 1]) == "D"
            and not (i > 1 and word[i - 1] in ALEFS and word[i - 2] == "ل")
        )
        if ch == "ل" and i + 1 < len(word) and word[i + 1] in ALEFS:
            out.append(
                (f"LIGATURE {LIGATURE_NAMES[word[i + 1]]}", "final" if prev_dual else "isolated")
            )
            i += 2
            continue
        right = prev_dual and joining(ch) in {"D", "R"}
        left = joining(ch) == "D" and i + 1 < len(word) and joining(word[i + 1]) in {"D", "R"}
        form = {
            (True, True): "medial",
            (True, False): "final",
            (False, True): "initial",
            (False, False): "isolated",
        }[(right, left)]
        out.append((unicodedata.name(ch).replace("ARABIC LETTER ", ""), form))
        i += 1
    return out


def actual_forms(shaped: str) -> list[tuple[str, str]]:
    out = []
    for ch in shaped:
        name = unicodedata.name(ch)
        form = form_of(ch)
        for suffix in (" ISOLATED FORM", " INITIAL FORM", " MEDIAL FORM", " FINAL FORM"):
            name = name.replace(suffix, "")
        name = name.replace("ARABIC LETTER ", "").replace("ARABIC ", "")
        out.append((name, "isolated" if form == "base" else form))
    return out


@pytest.mark.parametrize("unshaped_isolated", [False, True])
def test_l2_shaping_oracle(unshaped_isolated: bool) -> None:
    assert len(L2_WORDS) >= 30
    shaper = reshaper_for(unshaped_isolated)
    for word in L2_WORDS:
        shaped = shaper.reshape(word)
        assert actual_forms(shaped) == expected_forms(word), word
        if not unshaped_isolated:
            # no unshaped base letters remain in presentation-form mode
            assert all(form_of(c) != "base" for c in shaped), word


def test_l2_lam_alef_ligatures_all_contexts() -> None:
    shaper = reshaper_for(False)
    cases = {
        "لا": 0xFEFB,
        "كلا": 0xFEFC,
        "الأمل": 0xFEF7,
        "بلأ": 0xFEF8,
        "الإسلام": 0xFEF9,
        "بلإ": 0xFEFA,
        "الآن": 0xFEF5,
        "بلآ": 0xFEF6,
    }
    for word, cp in cases.items():
        assert chr(cp) in shaper.reshape(word), (word, hex(cp))


def test_l2_negative_control_detects_unshaped() -> None:
    assert actual_forms("محمد") != expected_forms("محمد")  # raw logical text


# ------------------------------------------------------------------- L3 bidi order
def test_l3_pure_arabic_is_reversed_shaping() -> None:
    for word in ("مرحبا بالعالم", "محمد", "الحمد لله", "بلا شك ولا تردد"):
        shaped = reshaper_for(False).reshape(word)
        assert shape_line(word) == shaped[::-1]


def test_l3_numbers_latin_brackets_and_terminal_punctuation() -> None:
    assert "10" in shape_line("انتظر 10 ثوان") and "01" not in shape_line("انتظر 10 ثوان")
    v = shape_line("اضغط على (OK) ثم")
    assert "(OK)" in v and ")OK(" not in v  # LTR run kept, brackets mirrored (L4)
    assert shape_line("«مرحبا»")[0] == "«" and shape_line("«مرحبا»")[-1] == "»"
    for text, mark in (("لا تذهب!", "!"), ("هل أنت بخير؟", "؟"), ("انتظر…", "…")):
        assert shape_line(text)[0] == mark  # sentence-final mark at the visual left end
    assert shape_line("العدد 12.5 و 3,000").count("12.5") == 1


def test_l3_base_direction_is_forced_rtl() -> None:
    # Auto-detection would make this LTR (first strong character is Latin).
    assert visual_order("OK هنا") == "انه OK"
    assert shape_line("5 تفاحات")[-1] == "5"  # the leading digit sits at the right end


def test_l3_multiline_wraps_logical_then_shapes_each_line() -> None:
    from manga_ar.config import load_config
    from manga_ar.detect.geometry import ellipse_mask
    from manga_ar.schemas import BBox, CropMask, Region, RegionType
    from manga_ar.typeset.layout import Typesetter

    text = CANONICAL[-1]
    box = BBox(0, 0, 260, 300)
    region = Region(
        id="r",
        type=RegionType.BUBBLE,
        bbox=box,
        fill_color=(255, 255, 255),
        bubble_mask=CropMask(box, ellipse_mask((300, 260), box)),
    )
    p = Typesetter(load_config(environ={}).typeset).layout(region, text, (300, 260))
    assert len(p.lines) >= 3
    from manga_ar.translate.normalize_ar import normalize_ar
    from manga_ar.typeset.arabic_text import strip_harakat

    expected_words = strip_harakat(normalize_ar(text)).split()
    assert " ".join(ln.logical for ln in p.lines).split() == expected_words  # order kept
    for ln in p.lines:
        assert ln.visual == p.engine.shape(ln.logical)  # each line shaped on its own
    # Negative control: shaping the paragraph first and then splitting reverses order.
    wrong_first = shape_line(" ".join(expected_words)).split()[:2]
    assert wrong_first != shape_line(p.lines[0].logical).split()[:2]


# ---------------------------------------------------------------- L4 glyph coverage
@pytest.mark.parametrize("face", BASIC_FONTS, ids=lambda f: f.key)
def test_l4_every_emitted_codepoint_is_covered(face: FontInfo) -> None:
    choice = FontChoice(face, SYMBOLS)
    for text in ALL_TEXTS:
        visual = choice.shape(text)
        runs = choice.runs(visual)
        for run, run_face in runs:
            for ch in run:
                assert ch.isspace() or ord(ch) in run_face.cmap, (face.key, text, hex(ord(ch)))
        assert not choice.missing, (face.key, text, choice.missing)


def test_l4_symbol_fallback_engages() -> None:
    choice = FontChoice(REG.get("NotoNaskhArabic"), SYMBOLS)
    for sym in "♡♪☆★♥":
        ((_run, face),) = choice.runs(sym)
        assert face.role == "symbol", sym
    bare = FontChoice(REG.get("NotoNaskhArabic"), [])
    assert bare.runs("♡") == [] and "♡" in bare.missing  # dropped + reported, never tofu


# ---------------------------------------------------------------------- L5 raster
def _render(draw_fn, text: str, size: int = 96) -> np.ndarray:  # type: ignore[no-untyped-def]
    img = Image.new("L", (size * (len(text) + 4), size * 3), 255)
    draw_fn(ImageDraw.Draw(img), (size, size // 2), text, size)
    a = np.asarray(img)
    ys, xs = np.nonzero(a < 200)
    return a[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def _components(a: np.ndarray) -> int:
    # Threshold 200 keeps faint anti-aliased joins (Cairo's hairline connections).
    return int(cv2.connectedComponents((a < 200).astype(np.uint8), connectivity=8)[0] - 1)


def _tall_on_right(a: np.ndarray) -> bool:
    """Direction oracle for "اب": alef (the only tall element; beh is low in every font)
    comes first logically, so in correct RTL output it must sit at the visual right."""
    ink = a < 200
    h = ink.shape[0]
    tops = np.array([np.argmax(col) if col.any() else h for col in ink.T])
    tall = np.flatnonzero(tops <= 0.3 * h)
    return bool(tall.size) and float(tall.mean()) > 0.6 * ink.shape[1]


def _paths(face: FontInfo):  # type: ignore[no-untyped-def]
    choice = FontChoice(face, [])
    good = lambda d, xy, t, s: choice.draw(d, xy, choice.shape(t), s, 0)  # noqa: E731
    unshaped = lambda d, xy, t, s: d.text(xy, t, font=load_font(str(face.path), s), fill=0)  # noqa: E731
    not_reordered = lambda d, xy, t, s: d.text(  # noqa: E731
        xy,
        reshaper_for(choice.unshaped_isolated).reshape(t),
        font=load_font(str(face.path), s),
        fill=0,
    )
    return good, unshaped, not_reordered


def _l5_ok(draw_fn) -> bool:  # type: ignore[no-untyped-def]
    return (
        _components(_render(draw_fn, "محمد")) == 1
        and _components(_render(draw_fn, "الحمد")) == 2
        and _tall_on_right(_render(draw_fn, "اب"))
    )


@pytest.mark.parametrize("face", BASIC_FONTS, ids=lambda f: f.key)
def test_l5_joining_and_direction_with_negative_controls(face: FontInfo) -> None:
    good, unshaped, not_reordered = _paths(face)
    assert _l5_ok(good), face.key
    assert not _l5_ok(unshaped), f"{face.key}: unshaped control passed"
    assert not _l5_ok(not_reordered), f"{face.key}: not-reordered control passed"


def test_l5_deterministic_render() -> None:
    good, _, _ = _paths(REG.get("NotoNaskhArabic"))
    assert np.array_equal(_render(good, CANONICAL[0]), _render(good, CANONICAL[0]))


# ------------------------------------------------------------ L6 cross-engine oracle
def _overlap(a: np.ndarray, b: np.ndarray) -> float:
    h, w = max(a.shape[0], b.shape[0]), max(a.shape[1], b.shape[1])
    ka = cv2.resize((a < 200).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    kb = cv2.resize((b < 200).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    da, db = cv2.dilate(ka, np.ones((3, 3), np.uint8)), cv2.dilate(kb, np.ones((3, 3), np.uint8))
    return float(((ka & db).sum() / max(1, ka.sum()) + (kb & da).sum() / max(1, kb.sum())) / 2)


# Calibrated on the canonical corpus words (DECISIONS D-030). Amiri's RAQM output uses
# contextual ligatures that presentation forms cannot express, hence its lower bar.
L6_THRESHOLDS = {"Amiri": 0.78}


@pytest.mark.skipif(not features.check("raqm"), reason="libraqm not available")
@pytest.mark.parametrize(
    "key",
    [
        "NotoNaskhArabic",
        "NotoSansArabic",
        "Amiri",
        "Cairo",
        "Tajawal",
        "Almarai",
        "Changa",
        "Lalezar",
    ],
)
def test_l6_basic_path_matches_raqm_oracle(key: str) -> None:
    face = REG.get(key)
    raqm = RaqmChoice(face)
    good, unshaped, not_reordered = _paths(face)
    oracle = lambda d, xy, t, s: raqm.draw(d, xy, raqm.shape(t), s, 0)  # noqa: E731
    words = sorted(
        {
            w
            for t in CANONICAL
            for w in t.split()
            if len(w) >= 2 and any("ء" <= c <= "ي" for c in w) and "♡" not in w
        }
    )
    scores = {"good": [], "unshaped": [], "not_reordered": []}
    for w in words:
        w = "".join(c for c in w if not "ً" <= c <= "ْ")
        ref = _render(oracle, w, 64)
        scores["good"].append(_overlap(_render(good, w, 64), ref))
        scores["unshaped"].append(_overlap(_render(unshaped, w, 64), ref))
        scores["not_reordered"].append(_overlap(_render(not_reordered, w, 64), ref))
    good_mean = float(np.mean(scores["good"]))
    neg = max(float(np.mean(scores["unshaped"])), float(np.mean(scores["not_reordered"])))
    assert good_mean >= L6_THRESHOLDS.get(key, 0.90), (key, good_mean)
    assert neg <= 0.70 and good_mean - neg >= 0.25, (key, good_mean, neg)
