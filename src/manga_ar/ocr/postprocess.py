"""OCR text normalisation and decoration extraction (S3 post-processing)."""

from __future__ import annotations

import re
import unicodedata

DECORATIONS = frozenset("♡♥♪♫☆★❤♢◇◆○●◎※")
_CJK_SPACE = re.compile(
    r"(?<=[぀-ヿ㐀-鿿豈-﫿＀-￯、。「」『』！？])\s+"
    r"(?=[぀-ヿ㐀-鿿豈-﫿＀-￯、。「」『』！？])"
)
_ELLIPSIS = re.compile(r"(?:\.{3,}|。{3,}|・{3,}|…+)")
_REPEAT_PUNCT = re.compile(r"([!?！？~〜ー])\1{3,}")


def extract_decorations(text: str) -> tuple[str, list[str]]:
    """Remove decorative symbols (♡ ♪ ☆ …) and return them in order of appearance."""
    found = [c for c in text if c in DECORATIONS]
    cleaned = "".join(c for c in text if c not in DECORATIONS)
    return cleaned, found


def normalize_ocr_text(text: str, lang: str | None) -> str:
    """NFKC (safe for CJK punctuation widths), whitespace and punctuation clean-up."""
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("　", " ")
    t = _ELLIPSIS.sub("…", t)
    t = t.replace("〜", "~").replace("～", "~")
    t = _REPEAT_PUNCT.sub(lambda m: m.group(1) * 3, t)
    if lang in {"ja", "zh"}:
        t = re.sub(r"\s+", "", t)  # no inter-word spaces in Japanese/Chinese
    else:
        t = _CJK_SPACE.sub("", t)
        t = re.sub(r"\s+", " ", t)
    return t.strip()


def postprocess(raw: str, lang: str | None, preserve_decorations: bool) -> tuple[str, list[str]]:
    text, decorations = extract_decorations(raw)
    text = normalize_ocr_text(text, lang)
    return text, (decorations if preserve_decorations else [])
