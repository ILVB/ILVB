"""Rendering of placements onto a transparent RGBA layer and compositing (A10)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFilter

from manga_ar.schemas import LayoutResult
from manga_ar.typeset.layout import Placement

RgbArray = npt.NDArray[np.uint8]
RgbaArray = npt.NDArray[np.uint8]


def render_layer(
    shape: tuple[int, int], placements: list[Placement], shadow: bool = False
) -> RgbaArray:
    """Draw every placement on a transparent RGBA layer of ``shape`` (h, w)."""
    h, w = shape
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    if shadow:
        shade = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        sdraw = ImageDraw.Draw(shade)
        for p in placements:
            off = max(1, p.size // 16)
            for line in p.lines:
                p.engine.draw(
                    sdraw,
                    (line.x + off, line.y + off),
                    line.visual,
                    p.size,
                    (0, 0, 0, 140),
                    p.outline_px,
                    (0, 0, 0, 140),
                )
        layer = Image.alpha_composite(layer, shade.filter(ImageFilter.GaussianBlur(1.5)))
    draw = ImageDraw.Draw(layer)
    for p in placements:
        fill = (*p.text_rgb, 255)
        stroke_fill = (*p.outline_rgb, 255) if p.outline_rgb is not None else None
        for line in p.lines:
            p.engine.draw(
                draw,
                (line.x, line.y),
                line.visual,
                p.size,
                fill,
                p.outline_px if stroke_fill is not None else 0,
                stroke_fill,
            )
    return np.asarray(layer, dtype=np.uint8).copy()


def composite(clean: RgbArray, layer: RgbaArray) -> RgbArray:
    """Alpha-composite the text layer over the cleaned page."""
    alpha = layer[..., 3:4].astype(np.float32) / 255.0
    out = clean.astype(np.float32) * (1.0 - alpha) + layer[..., :3].astype(np.float32) * alpha
    return np.asarray(np.rint(out).clip(0, 255), dtype=np.uint8)


def to_layout_result(p: Placement) -> LayoutResult:
    arabic = getattr(p.engine, "arabic", None)
    font = arabic.key if arabic is not None else "unknown"
    axes = getattr(p.engine, "axes", ())
    if axes:
        font += "@" + ",".join(f"{k}={v:g}" for k, v in axes)
    assert p.ink is not None
    return LayoutResult(
        font=font,
        size_px=p.size,
        lines=[ln.logical for ln in p.lines],
        strategy=p.strategy,
        box=p.ink,
        line_height=p.line_height,
        fill=p.text_rgb,
        outline=p.outline_rgb,
        outline_px=p.outline_px,
        ladder=list(p.ladder),
    )
