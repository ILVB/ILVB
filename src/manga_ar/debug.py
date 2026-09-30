"""Debug artifacts (``--debug``): detections, masks, reading order, inpaint diff, layout."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from manga_ar.io.writer import write_image
from manga_ar.schemas import Region, RegionType

RgbArray = npt.NDArray[np.uint8]
_COLORS = {
    RegionType.BUBBLE: (220, 40, 40),
    RegionType.NARRATION: (30, 140, 255),
    RegionType.FREE_TEXT: (20, 170, 60),
    RegionType.SFX: (190, 40, 190),
}


def detection_overlay(rgb: RgbArray, regions: list[Region]) -> RgbArray:
    """Bubble masks (tinted), text blocks, lines, safe boxes and reading order."""
    h, w = rgb.shape[:2]
    vis = rgb.astype(np.float32)
    for r in regions:
        if r.bubble_mask is not None:
            m = r.bubble_mask.to_full(h, w)
            vis[m] = vis[m] * 0.65 + np.array(_COLORS[r.type], np.float32) * 0.35
    out = np.clip(vis, 0, 255).astype(np.uint8)
    for r in regions:
        color = _COLORS[r.type]
        cv2.rectangle(out, (r.bbox.x0, r.bbox.y0), (r.bbox.x1 - 1, r.bbox.y1 - 1), color, 2)
        for ln in r.lines:
            cv2.rectangle(out, (ln.x0, ln.y0), (ln.x1 - 1, ln.y1 - 1), (255, 170, 0), 1)
        if r.safe_box is not None:
            sb = r.safe_box
            cv2.rectangle(out, (sb.x0, sb.y0), (sb.x1 - 1, sb.y1 - 1), (0, 200, 200), 1)
        label = f"{r.reading_order}{'V' if r.vertical else 'H'}"
        cv2.putText(
            out,
            label,
            (r.bbox.x0, max(14, r.bbox.y0 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def mask_overlay(rgb: RgbArray, regions: list[Region], which: str) -> RgbArray:
    """``which`` in {"text", "inpaint"}: paint that mask in red over a dimmed page."""
    h, w = rgb.shape[:2]
    vis = (rgb.astype(np.float32) * 0.5 + 127).astype(np.uint8)
    for r in regions:
        mask = r.text_mask if which == "text" else r.inpaint_mask
        if mask is not None:
            vis[mask.to_full(h, w)] = (230, 20, 20)
    return vis


def inpaint_diff(before: RgbArray, after: RgbArray) -> RgbArray:
    """Heat-map of per-pixel change (black = untouched)."""
    diff = np.abs(before.astype(np.int16) - after.astype(np.int16)).max(axis=2).astype(np.uint8)
    heat = cv2.applyColorMap(cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    heat[diff == 0] = 0
    return np.asarray(heat, dtype=np.uint8)


def layout_overlay(rgb: RgbArray, regions: list[Region]) -> RgbArray:
    out = rgb.copy()
    for r in regions:
        if r.layout is None:
            continue
        b = r.layout.box
        cv2.rectangle(out, (b.x0, b.y0), (b.x1 - 1, b.y1 - 1), (0, 160, 255), 1)
        cv2.putText(
            out,
            f"{r.layout.size_px}px {r.layout.strategy}",
            (b.x0, max(12, b.y0 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 120, 220),
            1,
            cv2.LINE_AA,
        )
    return out


def write_debug(directory: Path, name: str, image: RgbArray) -> Path:
    path = directory / f"{name}.png"
    write_image(path, image, "png")
    return path
