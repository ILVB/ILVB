"""Bubble segmentation, region typing and block merging (S2).

For each text block the bubble is grown by flood fill from seeds around the text (the
glyphs are painted out first so the fill passes over them). Floating-range flood fill
follows gradients but stops at outlines. Implausible results (leaks) fall back to an
ellipse fitted around the text and are flagged ``LEAK_FALLBACK``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from manga_ar.config import DetectConfig
from manga_ar.detect.base import TextBlock
from manga_ar.detect.geometry import (
    disk,
    ellipse_mask,
    fill_holes,
    largest_component,
    largest_inscribed_rect,
    mask_polygon,
    solidity,
)
from manga_ar.logging_setup import get_logger
from manga_ar.schemas import BBox, CropMask, Flag, Region, RegionType

log = get_logger(__name__)
U8 = npt.NDArray[np.uint8]
BoolArray = npt.NDArray[np.bool_]


@dataclass
class _PageCtx:
    """Per-page state shared by the growth of every block."""

    all_text: BoolArray  # union of all blocks' glyph pixels (full page)
    blocks: list[TextBlock]


@dataclass
class _Grown:
    mask: BoolArray | None  # full-ROI mask of the bubble interior (None = leak)
    roi: BBox
    reason: str = ""
    raw: BoolArray | None = None  # interior before tail-severing (keeps sharp corners)


def inscribed_safe_box(mask: BoolArray, offset: tuple[int, int], erode: int = 2) -> BBox | None:
    """Largest inscribed rectangle of ``mask`` eroded by ``erode`` px (page coords).

    Runs on a ≤ 160 px downsampled copy for speed. The result is shrunk back into the
    full-resolution eroded mask, so it never exceeds the true interior.
    """
    m = cv2.erode(mask.astype(np.uint8), disk(erode)) > 0 if erode > 0 else mask
    if not m.any():
        return None
    h, w = m.shape
    scale = max(1.0, max(h, w) / 160.0)
    small = (
        cv2.resize(
            m.astype(np.uint8),
            (max(1, int(w / scale)), max(1, int(h / scale))),
            interpolation=cv2.INTER_AREA,
        )
        >= 1
    )  # conservative: only full cells
    rect = largest_inscribed_rect(small)
    if rect is None:
        return None
    x0, y0 = int(np.ceil(rect.x0 * scale)), int(np.ceil(rect.y0 * scale))
    x1, y1 = int(np.floor(rect.x1 * scale)), int(np.floor(rect.y1 * scale))
    x1, y1 = min(x1, w), min(y1, h)
    # shrink until fully inside the full-resolution mask
    for _ in range(64):
        if x1 <= x0 or y1 <= y0:
            return None
        if m[y0:y1, x0:x1].all():
            break
        x0, y0, x1, y1 = x0 + 1, y0 + 1, x1 - 1, y1 - 1
    else:
        return None
    return BBox(x0 + offset[0], y0 + offset[1], x1 + offset[0], y1 + offset[1])


class FloodBubbleSegmenter:
    """Default bubble segmenter (classical, deterministic)."""

    def __init__(self, cfg: DetectConfig) -> None:
        self.cfg = cfg

    def segment(self, rgb: U8, blocks: list[TextBlock], page_id: str) -> list[Region]:
        h, w = rgb.shape[:2]
        text_blocks = [b for b in blocks if not b.is_sfx]
        all_text = np.zeros((h, w), bool)
        for b in text_blocks:
            if b.text_mask is not None:
                all_text |= b.text_mask.to_full(h, w)
        ctx = _PageCtx(all_text, text_blocks)
        regions: list[Region] = []
        for i, block in enumerate(blocks):
            region = self._region_for(rgb, block, f"{page_id}-r{i}", ctx)
            regions.append(region)
        return self._merge_shared_bubbles(regions, h, w)

    # ---------------------------------------------------------------- core
    def _region_for(self, rgb: U8, block: TextBlock, rid: str, ctx: _PageCtx) -> Region:
        h, w = rgb.shape[:2]
        region = Region(
            id=rid,
            type=RegionType.SFX if block.is_sfx else RegionType.BUBBLE,
            bbox=block.bbox,
            lines=list(block.lines),
            text_mask=block.text_mask,
            vertical=block.vertical,
            score=block.score,
        )
        if block.is_sfx:
            region.polygon = _rect_polygon(block.bbox)
            region.safe_box = block.bbox
            return region
        grown = self._grow(rgb, block, ctx, scale=self.cfg.bubble_roi_scale)
        for factor in (1.8, 3.2):  # large bubbles around little text need a bigger window
            if grown.mask is not None or grown.reason != "roi":
                break
            grown = self._grow(rgb, block, ctx, scale=self.cfg.bubble_roi_scale * factor)
        fill = self._local_fill(rgb, block)
        region.fill_color = fill
        luminance = 0.299 * fill[0] + 0.587 * fill[1] + 0.114 * fill[2]
        if grown.mask is None:
            if luminance >= 225 and self._uniform_around(rgb, block):
                # A white bubble whose outline is open (touches a gutter / panel border):
                # keep it a bubble but use a conservative ellipse around the text.
                region.type = RegionType.BUBBLE
                region.flag(Flag.LEAK_FALLBACK)
                self._fallback_shape(region, block, h, w)
            else:
                region.type = RegionType.FREE_TEXT
                pad = max(2, int(0.15 * block.glyph_size))
                box = block.bbox.expand(pad).clip(w, h)
                region.polygon = _rect_polygon(box)
                region.safe_box = box
            log.debug("%s: bubble growth rejected (%s)", rid, grown.reason)
            return region
        roi = grown.roi
        mask = grown.mask
        region.bubble_mask = CropMask.from_full(mask, offset=(roi.x0, roi.y0))
        region.polygon = [(x + roi.x0, y + roi.y0) for x, y in mask_polygon(mask)]
        region.safe_box = inscribed_safe_box(mask, (roi.x0, roi.y0))
        if self._is_rectangular(grown.raw if grown.raw is not None else mask):
            region.type = RegionType.NARRATION
        return region

    def _grow(self, rgb: U8, block: TextBlock, ctx: _PageCtx, scale: float) -> _Grown:
        h, w = rgb.shape[:2]
        bb = block.bbox
        margin = int(max(bb.width, bb.height) * (scale - 1.0) / 2.0) + 12
        roi = bb.expand(margin).clip(w, h)
        crop = rgb[roi.y0 : roi.y1, roi.x0 : roi.x1].copy()
        text = (
            block.text_mask.window(roi)
            if block.text_mask is not None
            else np.zeros((roi.height, roi.width), bool)
        )
        # every block's glyphs inside the window are painted out, not only this block's
        others = ctx.all_text[roi.y0 : roi.y1, roi.x0 : roi.x1]
        text_d = cv2.dilate((text | others).astype(np.uint8), disk(2)) > 0
        ring = (cv2.dilate(text.astype(np.uint8), disk(6)) > 0) & ~text_d
        if not ring.any():
            return _Grown(None, roi, "no-ring")
        med = np.median(crop[ring], axis=0).astype(np.uint8)
        crop[text_d] = med  # paint glyphs out so the fill crosses them
        ff = np.zeros((roi.height + 2, roi.width + 2), np.uint8)
        flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
        lo = up = (5, 5, 5)
        close = np.abs(crop.astype(np.int16) - med.astype(np.int16)).max(axis=2) <= 12
        seeds = np.argwhere(ring & close)
        if seeds.size == 0:
            return _Grown(None, roi, "no-seed")
        step = max(1, len(seeds) // 24)
        bgr = np.ascontiguousarray(crop[..., ::-1])
        for sy, sx in seeds[::step]:
            if ff[sy + 1, sx + 1]:
                continue
            cv2.floodFill(bgr, ff, (int(sx), int(sy)), (0, 0, 0), lo, up, flags)
        grown = ff[1:-1, 1:-1] > 0
        grown |= text
        # sever tails and thin leaks, keep the component holding the text
        k = max(2, round(0.25 * block.glyph_size))
        opened = cv2.morphologyEx(grown.astype(np.uint8), cv2.MORPH_OPEN, disk(k)) > 0
        opened |= text
        body = self._component_with_text(opened, text)
        body = fill_holes(body)
        result = self._validate(body, text, roi, (h, w), ctx)
        result.raw = fill_holes(self._component_with_text(grown, text))
        return result

    @staticmethod
    def _component_with_text(mask: BoolArray, text: BoolArray) -> BoolArray:
        n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        if n <= 1:
            return mask
        ids = labels[text & mask]
        if ids.size == 0:
            return largest_component(mask)
        keep = int(np.bincount(ids[ids > 0]).argmax()) if (ids > 0).any() else 0
        return np.asarray(labels == keep, dtype=np.bool_)

    def _validate(
        self, mask: BoolArray, text: BoolArray, roi: BBox, shape: tuple[int, int], ctx: _PageCtx
    ) -> _Grown:
        h, w = shape
        area = int(mask.sum())
        tbox = BBox.from_mask(text)
        text_area = tbox.area if tbox is not None else 1
        # a bubble may hold several blocks: count every block whose centre lies inside it
        for other in ctx.blocks:
            cx, cy = (int(v) for v in other.bbox.center)
            lx, ly = cx - roi.x0, cy - roi.y0
            if tbox is not None and other.bbox.iou(tbox.translate(roi.x0, roi.y0)) > 0.9:
                continue
            if 0 <= lx < roi.width and 0 <= ly < roi.height and mask[ly, lx]:
                text_area += other.bbox.area
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            return _Grown(None, roi, "empty")
        touches_roi = (
            (ys.min() == 0 and roi.y0 > 0)
            or (xs.min() == 0 and roi.x0 > 0)
            or (ys.max() == roi.height - 1 and roi.y1 < h)
            or (xs.max() == roi.width - 1 and roi.x1 < w)
        )
        touches_page = (
            (roi.y0 == 0 and ys.min() == 0)
            or (roi.x0 == 0 and xs.min() == 0)
            or (roi.y1 == h and ys.max() == roi.height - 1)
            or (roi.x1 == w and xs.max() == roi.width - 1)
        )
        if touches_roi:
            return _Grown(None, roi, "roi")
        if touches_page:
            return _Grown(None, roi, "page-border")
        if area > self.cfg.leak_max_area_ratio * text_area:
            return _Grown(None, roi, "area-ratio")
        if area > self.cfg.leak_max_page_fraction * h * w:
            return _Grown(None, roi, "page-fraction")
        if solidity(mask) < self.cfg.leak_min_solidity:
            return _Grown(None, roi, "solidity")
        if area < 1.05 * text_area:
            return _Grown(None, roi, "no-margin")  # no background around text: free text
        return _Grown(mask, roi)

    @staticmethod
    def _is_rectangular(mask: BoolArray) -> bool:
        box = BBox.from_mask(mask)
        if box is None:
            return False
        sub = mask[box.y0 : box.y1, box.x0 : box.x1]
        rect = float(sub.mean())
        c = max(2, min(box.width, box.height) // 25)
        corners = [sub[:c, :c], sub[:c, -c:], sub[-c:, :c], sub[-c:, -c:]]
        return rect >= 0.97 and all(float(k.mean()) >= 0.8 for k in corners)

    @staticmethod
    def _local_fill(rgb: U8, block: TextBlock) -> tuple[int, int, int]:
        h, w = rgb.shape[:2]
        roi = block.bbox.expand(int(0.5 * block.glyph_size) + 6).clip(w, h)
        text = (
            block.text_mask.window(roi)
            if block.text_mask is not None
            else np.zeros((roi.height, roi.width), bool)
        )
        near = cv2.dilate(text.astype(np.uint8), disk(max(3, int(0.3 * block.glyph_size)))) > 0
        ring = near & ~(cv2.dilate(text.astype(np.uint8), disk(2)) > 0)
        crop = rgb[roi.y0 : roi.y1, roi.x0 : roi.x1]
        vals = crop[ring] if ring.any() else crop.reshape(-1, 3)
        med = np.median(vals, axis=0)
        return (int(med[0]), int(med[1]), int(med[2]))

    @staticmethod
    def _uniform_around(rgb: U8, block: TextBlock) -> bool:
        h, w = rgb.shape[:2]
        roi = block.bbox.expand(int(0.4 * block.glyph_size) + 4).clip(w, h)
        text = (
            block.text_mask.window(roi)
            if block.text_mask is not None
            else np.zeros((roi.height, roi.width), bool)
        )
        ring = ~(cv2.dilate(text.astype(np.uint8), disk(2)) > 0)
        gray = cv2.cvtColor(rgb[roi.y0 : roi.y1, roi.x0 : roi.x1], cv2.COLOR_RGB2GRAY)
        vals = gray[ring]
        return bool(vals.size and float(vals.std()) <= 12.0)

    def _fallback_shape(self, region: Region, block: TextBlock, h: int, w: int) -> None:
        pad = max(4, int(0.45 * block.glyph_size))
        # an ellipse circumscribing the padded text box (√2 factor) but clipped to the page
        bb = block.bbox.expand(pad)
        cx, cy = bb.center
        ew, eh = bb.width * 1.35, bb.height * 1.35
        ebox = BBox(int(cx - ew / 2), int(cy - eh / 2), int(cx + ew / 2), int(cy + eh / 2))
        ebox = ebox.clip(w, h)
        mask = ellipse_mask((ebox.height, ebox.width), BBox(0, 0, ebox.width, ebox.height))
        region.bubble_mask = CropMask(ebox, mask)
        region.polygon = [(x + ebox.x0, y + ebox.y0) for x, y in mask_polygon(mask)]
        region.safe_box = inscribed_safe_box(mask, (ebox.x0, ebox.y0))

    # --------------------------------------------------------------- merge
    @staticmethod
    def _merge_shared_bubbles(regions: list[Region], h: int, w: int) -> list[Region]:
        """Blocks whose text lies inside another block's bubble become one region."""
        merged: list[Region] = []
        consumed: set[str] = set()
        by_area = sorted(
            regions, key=lambda r: -(r.bubble_mask.area if r.bubble_mask is not None else 0)
        )
        for host in by_area:
            if host.id in consumed:
                continue
            if (
                host.bubble_mask is not None
                and host.type
                in {
                    RegionType.BUBBLE,
                    RegionType.NARRATION,
                }
                and Flag.LEAK_FALLBACK not in host.flags
            ):
                full = host.bubble_mask
                for other in regions:
                    if other is host or other.id in consumed or other.type == RegionType.SFX:
                        continue
                    cx, cy = (int(v) for v in other.bbox.center)
                    inside = full.bbox.x0 <= cx < full.bbox.x1 and full.bbox.y0 <= cy < full.bbox.y1
                    if inside and full.data[cy - full.bbox.y0, cx - full.bbox.x0]:
                        _absorb(host, other)
                        consumed.add(other.id)
            merged.append(host)
        merged = [r for r in merged if r.id not in consumed]
        merged.sort(key=lambda r: int(r.id.rsplit("r", 1)[-1]))
        return merged


def _absorb(host: Region, other: Region) -> None:
    host.bbox = host.bbox.union(other.bbox)
    host.lines = host.lines + other.lines
    if host.vertical:
        host.lines.sort(key=lambda b: -b.x1)
    else:
        host.lines.sort(key=lambda b: (b.y0, b.x0))
    if host.text_mask is not None and other.text_mask is not None:
        box = host.text_mask.bbox.union(other.text_mask.bbox)
        data = host.text_mask.window(box) | other.text_mask.window(box)
        host.text_mask = CropMask(box, data)
    host.flag(Flag.MERGED)


def _rect_polygon(box: BBox) -> list[tuple[int, int]]:
    return [(box.x0, box.y0), (box.x1, box.y0), (box.x1, box.y1), (box.x0, box.y1)]
