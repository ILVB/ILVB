"""Context-aware batching: all regions of a page in one index-tagged request (S5).

Segments are sent as ``[1] text [2] text …`` on one line (the Google web endpoint
collapses newlines). The response is parsed back into exactly N segments; any missing,
duplicated or out-of-order marker makes the caller fall back to per-region requests.
"""

from __future__ import annotations

import re

_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹０１２３４５６７８９", "0123456789" * 3)
_MARKER = re.compile(r"[\[［【〔(（]\s*([0-9٠-٩۰-۹０-９]{1,3})\s*[\]］】〕)）]")


def encode(texts: list[str]) -> str:
    return " ".join(f"[{i + 1}] {t}" for i, t in enumerate(texts))


def decode(response: str, n: int) -> list[str] | None:
    """Split a tagged response into ``n`` segments, or ``None`` if integrity fails."""
    matches = list(_MARKER.finditer(response))
    indices = [int(m.group(1).translate(_DIGITS)) for m in matches]
    if indices != list(range(1, n + 1)):
        return None
    if response[: matches[0].start()].strip():
        return None  # text before the first marker: the segmentation shifted
    segments = []
    for k, m in enumerate(matches):
        end = matches[k + 1].start() if k + 1 < len(matches) else len(response)
        seg = response[m.end() : end].strip()
        if not seg:
            return None
        segments.append(seg)
    return segments


def chunk(texts: list[str], limit: int) -> list[list[int]]:
    """Group indices so each encoded request stays within ``limit`` characters.

    A single text longer than the limit forms its own chunk (the provider call then
    fails validation and the per-region path handles it).
    """
    chunks: list[list[int]] = []
    current: list[int] = []
    size = 0
    for i, t in enumerate(texts):
        cost = len(f"[{len(current) + 1}] ") + len(t) + 1
        if current and size + cost > limit:
            chunks.append(current)
            current, size = [], 0
            cost = len("[1] ") + len(t) + 1
        current.append(i)
        size += cost
    if current:
        chunks.append(current)
    return chunks
