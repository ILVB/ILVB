"""Inpaint-mask construction (S4): text pixels, adaptive dilation, allowed zone."""

from __future__ import annotations

import cv2
import numpy as np

from manga_ar.config import InpaintConfig
from manga_ar.detect.geometry import disk, erode
from manga_ar.inpaint.base import BoolArray, RgbArray
from manga_ar.schemas import BBox, CropMask, Flag, Region, RegionType


def stroke_width(text: BoolArray) -> float:
    """Approximate stroke width: twice the 90th-percentile distance to the background."""
    if not text.any():
        return 0.0
    # pad with background so a mask filling its whole (tight) crop still has a zero pixel
    padded = np.pad(text.astype(np.uint8), 1)
    dt = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
    return float(2.0 * np.percentile(dt[text], 90))


def threshold_text(rgb: RgbArray, box: BBox) -> BoolArray:
    """Fallback text mask when the detector gave none: polarity-aware Otsu in ``box``."""
    gray = cv2.cvtColor(rgb[box.y0 : box.y1, box.x0 : box.x1], cv2.COLOR_RGB2GRAY)
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    dark_text = float(np.median(border)) >= 128
    flag = cv2.THRESH_BINARY_INV if dark_text else cv2.THRESH_BINARY
    _, binary = cv2.threshold(gray, 0, 255, flag | cv2.THRESH_OTSU)
    return np.asarray(binary > 0, dtype=np.bool_)


def allowed_zone(
    region: Region, shape: tuple[int, int], cfg: InpaintConfig, dilate: int
) -> CropMask:
    """Where inpainting may write: the bubble interior eroded by ``allowed_erode`` (outline
    and tail survive), or around the text itself for free text / fallback shapes."""
    h, w = shape
    use_bubble = (
        region.bubble_mask is not None
        and region.type in {RegionType.BUBBLE, RegionType.NARRATION}
        and Flag.LEAK_FALLBACK not in region.flags
    )
    if use_bubble:
        bm = region.bubble_mask
        assert bm is not None
        data = bm.data
        if cfg.allowed_erode > 0:
            data = erode(data, cfg.allowed_erode)
        return CropMask(bm.bbox, np.asarray(data, dtype=np.bool_))
    # Free text / leaked bubble: the dilated text itself plus a halo margin.
    base = region.text_mask
    glyph = (
        float(np.median([min(b.width, b.height) for b in region.lines])) if region.lines else 0.0
    )
    margin = dilate + 3 + (round(0.45 * glyph) if region.type == RegionType.FREE_TEXT else 0)
    if base is None:
        box = region.bbox.expand(margin).clip(w, h)
        return CropMask(box, np.ones((box.height, box.width), bool))
    box = base.bbox.expand(margin).clip(w, h)
    grown = cv2.dilate(base.window(box).astype(np.uint8), disk(margin)) > 0
    return CropMask(box, np.asarray(grown, dtype=np.bool_))


def halo_mask(rgb: RgbArray, text: CropMask, glyph: float) -> CropMask | None:
    """Outline/halo drawn around free text (e.g. white stroke over artwork).

    Pixels within ~0.45 glyph of the text that share the colour hugging the glyphs are
    taken as halo when that colour contrasts with the background further out.
    """
    h, w = rgb.shape[:2]
    reach = max(3, round(0.45 * glyph))
    box = text.bbox.expand(reach + 12).clip(w, h)
    t = text.window(box).astype(np.uint8)
    crop = rgb[box.y0 : box.y1, box.x0 : box.x1].astype(np.int16)
    hug = (cv2.dilate(t, disk(3)) > 0) & ~(cv2.dilate(t, disk(1)) > 0)
    far = (cv2.dilate(t, disk(reach + 10)) > 0) & ~(cv2.dilate(t, disk(reach + 4)) > 0)
    if not hug.any() or not far.any():
        return None
    hug_c = np.median(crop[hug], axis=0)
    far_c = np.median(crop[far], axis=0)
    if np.abs(hug_c - far_c).max() < 60:
        return None  # no contrasting halo
    similar = np.abs(crop - hug_c).max(axis=2) <= 40
    zone = cv2.dilate(t, disk(reach)) > 0
    candidate = (similar & zone) | (t > 0)
    _, labels = cv2.connectedComponents(candidate.astype(np.uint8), connectivity=8)
    keep = np.isin(labels, np.unique(labels[t > 0]))
    keep &= labels > 0
    return CropMask.from_full(keep & ~(t > 0), offset=(box.x0, box.y0))


def stray_ink(rgb: RgbArray, text: CropMask, glyph: float) -> CropMask | None:
    """Pixels near the text that deviate from the local background (anti-aliased stroke
    tips, dots and dakuten the detector's glyph mask missed). Only used on bubble and
    caption interiors, where anything that is not background around the text is text."""
    h, w = rgb.shape[:2]
    reach = max(3, round(0.35 * glyph))
    box = text.bbox.expand(reach + 2).clip(w, h)
    gray = cv2.cvtColor(rgb[box.y0 : box.y1, box.x0 : box.x1], cv2.COLOR_RGB2GRAY)
    k = max(15, 2 * round(0.6 * glyph) + 1)
    background = cv2.medianBlur(gray, k)
    deviates = np.abs(gray.astype(np.int16) - background.astype(np.int16)) > 30
    near = cv2.dilate(text.window(box).astype(np.uint8), disk(reach)) > 0
    found = deviates & near
    return CropMask.from_full(found, offset=(box.x0, box.y0)) if found.any() else None


def build_mask(
    rgb: RgbArray, region: Region, cfg: InpaintConfig, extra_dilate: int = 0
) -> CropMask | None:
    """Final inpaint mask for ``region`` (page coordinates), or ``None`` if empty."""
    h, w = rgb.shape[:2]
    if region.text_mask is not None and region.text_mask.area > 0:
        text = region.text_mask
    else:
        box = region.bbox.clip(w, h)
        if box.area == 0:
            return None
        text = CropMask(box, threshold_text(rgb, box))
    sw = stroke_width(text.data)
    dilate = int(
        np.clip(round(0.5 * sw + 1) + extra_dilate, cfg.dilate_min, cfg.dilate_max + extra_dilate)
    )
    glyph = (
        float(np.median([min(b.width, b.height) for b in region.lines]))
        if region.lines
        else 2.5 * sw
    )
    if region.type in {RegionType.BUBBLE, RegionType.NARRATION}:
        stray = stray_ink(rgb, text, glyph)
        if stray is not None:
            union = text.bbox.union(stray.bbox)
            text = CropMask(union, text.window(union) | stray.window(union))
    if region.type == RegionType.FREE_TEXT:
        halo = halo_mask(rgb, text, glyph)
        if halo is not None:
            union = text.bbox.union(halo.bbox)
            text = CropMask(union, text.window(union) | halo.window(union))
    zone = allowed_zone(region, (h, w), cfg, dilate)
    box = text.bbox.expand(dilate + 2).clip(w, h)
    grown = cv2.dilate(text.window(box).astype(np.uint8), disk(dilate)) > 0
    final = grown & zone.window(box)
    if not final.any():
        return None
    return CropMask.from_full(final, offset=(box.x0, box.y0))
