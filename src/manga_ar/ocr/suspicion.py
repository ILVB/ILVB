"""OCR result validation (``OCR_SUSPECT``): catch garbage before it reaches translation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from manga_ar.ocr.langid import script_consistency

_LOOP = re.compile(r"(.{1,4}?)\1{4,}")


@dataclass
class Verdict:
    suspicious: bool
    score: float  # plausibility in [0, 1]; higher is better
    reasons: list[str] = field(default_factory=list)


def assess(
    text: str,
    lang: str,
    confidence: float | None,
    area_px: int,
    glyph_px: float,
    min_confidence: float,
) -> Verdict:
    """Heuristic plausibility of an OCR result.

    Checks: empty text, script consistency with ``lang``, repetition loops (typical
    hallucination on non-text), characters-per-area density and engine confidence.
    """
    reasons: list[str] = []
    score = 1.0
    stripped = "".join(text.split())
    if not stripped:
        return Verdict(True, 0.0, ["empty"])
    consistency = script_consistency(stripped, lang)
    if consistency < 0.5:
        reasons.append(f"script-mismatch({consistency:.2f})")
        score *= max(0.05, consistency)
    if _LOOP.search(stripped):
        reasons.append("repetition-loop")
        score *= 0.1
    if glyph_px > 0 and area_px > 0:
        capacity = area_px / (glyph_px * glyph_px)  # glyphs that fit in the region
        n = len(stripped)
        if n > 2.5 * capacity + 3:
            reasons.append(f"too-dense({n}>{capacity:.1f})")
            score *= 0.3
        elif capacity >= 6 and n < 0.12 * capacity:
            reasons.append(f"too-sparse({n}<<{capacity:.1f})")
            score *= 0.6
    if confidence is not None:
        if confidence < min_confidence:
            reasons.append(f"low-confidence({confidence:.2f})")
        score *= 0.5 + 0.5 * max(0.0, min(1.0, confidence))
    suspicious = any(not r.startswith("too-sparse") for r in reasons)
    return Verdict(suspicious, round(score, 4), reasons)
