"""Non-destructive OCR preprocessing (v2, step 1.0.4): texture suppression on OCR crops.

Only the image handed to the recogniser changes; the page is never modified. On a crop
whose background is dense with small marks (screentone dots), the glyph strokes are kept
(Otsu ink, polarity-aware), along with small marks next to a glyph (punctuation). Everything
else becomes flat background. Crops without that texture are returned unchanged.
"""

from __future__ import annotations

import cv2
import numpy as np

from manga_ar.ocr.base import RgbArray

DOT_DENSITY = 15.0  # small marks per 1000 px above which the background counts as textured
MAX_DOT_AREA = 40  # px; screentone dots are at most this large (at page resolution)
NEAR = 0.35  # small marks within NEAR x line height of a glyph stroke are kept


def _dot_density(gray: np.ndarray, threshold: float) -> float:
    light = ((gray < np.percentile(gray, 99) - 12) & (gray > threshold)).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(light, connectivity=8)
    small = int(((stats[1:, 4] >= 2) & (stats[1:, 4] <= MAX_DOT_AREA)).sum()) if n > 1 else 0
    return small / (gray.size / 1000.0)


def suppress_texture(crop: RgbArray) -> RgbArray:
    """The crop with screentone-like texture removed, or the crop itself if it has none."""
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    if min(gray.shape) < 8:
        return crop
    invert = float(np.median(gray)) < 110  # light text on a dark background
    g = 255 - gray if invert else gray
    threshold, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if _dot_density(g, threshold) < DOT_DENSITY:
        return crop
    ink = (g <= threshold).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n <= 1:
        return crop
    line_h = g.shape[0]
    big = np.zeros(n, bool)
    big[1:] = np.maximum(stats[1:, 2], stats[1:, 3]) >= 0.25 * line_h
    strokes = big[labels]
    radius = max(1, round(NEAR * line_h))
    near = cv2.dilate(strokes.astype(np.uint8), np.ones((2 * radius + 1,) * 2, np.uint8)) > 0
    keep_ids = np.unique(labels[near & (ink > 0)])
    keep = np.isin(labels, keep_ids[keep_ids > 0])
    keep = cv2.dilate(keep.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0  # anti-aliasing
    clean = np.full_like(g, 255)
    clean[keep] = np.minimum(g[keep], 255)
    out = 255 - clean if invert else clean
    return np.repeat(out[..., None], 3, axis=2)
