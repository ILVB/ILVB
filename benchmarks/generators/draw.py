"""Drawing primitives: textures, bubbles with tails, and text blocks with exact alpha."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont

from benchmarks.generators.fonts import font_path

U8 = npt.NDArray[np.uint8]
Bool = npt.NDArray[np.bool_]


# ------------------------------------------------------------------ textures
def gradient(h: int, w: int, c0: int, c1: int, vertical: bool = True) -> U8:
    ramp = np.linspace(c0, c1, h if vertical else w, dtype=np.float64)
    grid = np.repeat(ramp[:, None], w, axis=1) if vertical else np.repeat(ramp[None, :], h, axis=0)
    return np.repeat(np.rint(grid).astype(np.uint8)[..., None], 3, axis=2)


def screentone(h: int, w: int, rng: np.random.Generator, bg: int = 245, fg: int = 90) -> U8:
    """Halftone dots on a rotated grid (manga screentone)."""
    spacing = float(rng.uniform(5.0, 8.0))
    radius = float(rng.uniform(1.0, 2.2))
    angle = math.radians(float(rng.choice([0.0, 30.0, 45.0])))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    u = xx * math.cos(angle) + yy * math.sin(angle)
    v = -xx * math.sin(angle) + yy * math.cos(angle)
    du = (u % spacing) - spacing / 2
    dv = (v % spacing) - spacing / 2
    dots = du * du + dv * dv <= radius * radius
    out = np.full((h, w), bg, np.uint8)
    out[dots] = fg
    return np.repeat(out[..., None], 3, axis=2)


def artwork(h: int, w: int, rng: np.random.Generator) -> U8:
    """Busy background: gradient + hatching + shapes (stands in for drawn art)."""
    img = gradient(
        h, w, int(rng.integers(150, 235)), int(rng.integers(150, 235)), bool(rng.random() < 0.5)
    )
    canvas = Image.fromarray(img)
    draw = ImageDraw.Draw(canvas)
    step = int(rng.integers(6, 14))
    tone = int(rng.integers(60, 130))
    for x in range(-h, w, step):
        draw.line([(x, 0), (x + h, h)], fill=(tone, tone, tone), width=1)
    for _ in range(int(rng.integers(3, 8))):
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(10, max(12, min(h, w) // 3)))
        g = int(rng.integers(40, 200))
        draw.ellipse(
            [x0 - r, y0 - r, x0 + r, y0 + r], outline=(g, g, g), width=int(rng.integers(2, 6))
        )
    return np.asarray(canvas, dtype=np.uint8).copy()


# ------------------------------------------------------------------ bubbles
def body_mask(shape: str, w: int, h: int, rng: np.random.Generator) -> Bool:
    """Filled bubble body of size (h, w): ellipse, roundrect, spiky or rect (captions)."""
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    if shape == "ellipse":
        draw.ellipse([0, 0, w - 1, h - 1], fill=255)
    elif shape == "roundrect":
        draw.rounded_rectangle([0, 0, w - 1, h - 1], radius=int(min(w, h) * 0.25), fill=255)
    elif shape == "rect":
        draw.rectangle([0, 0, w - 1, h - 1], fill=255)
    elif shape == "spiky":
        n = int(rng.integers(14, 22))
        pts = []
        for k in range(2 * n):
            a = math.pi * k / n
            rr = 1.0 if k % 2 == 0 else float(rng.uniform(0.78, 0.86))
            pts.append(
                (w / 2 + rr * (w / 2 - 1) * math.cos(a), h / 2 + rr * (h / 2 - 1) * math.sin(a))
            )
        draw.polygon(pts, fill=255)
    else:
        raise ValueError(shape)
    return np.asarray(img) > 0


def tail_polygon(box: tuple[int, int, int, int], rng: np.random.Generator) -> list[tuple[int, int]]:
    """A tail from the lower half of the bubble outward (page coordinates)."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = 1 if rng.random() < 0.5 else -1
    base = (cx + side * (x1 - x0) * 0.15, cy + (y1 - y0) * 0.35)
    tip = (base[0] + side * rng.uniform(15, 45), y1 + rng.uniform(20, 50))
    half = (x1 - x0) * 0.08
    return [
        (int(base[0] - half), int(base[1])),
        (int(base[0] + half), int(base[1])),
        (int(tip[0]), int(tip[1])),
    ]


# ------------------------------------------------------------------ text
@lru_cache(maxsize=256)
def _font(key: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(font_path(key)), size)


@dataclass
class TextBlock:
    rgba: U8  # (h, w, 4) text layer, alpha = coverage
    lines: list[str]


# Characters that must not start a line or column (kinsoku), as in real CJK lettering.
NO_LINE_START = frozenset("。、，．！？!?…‥・ー）」』〉》】,.:;")


def wrap(text: str, key: str, size: int, max_width: int, lang: str) -> list[str]:
    font = _font(key, size)
    units = text.split(" ") if lang in ("en", "ko") else list(text)
    sep = " " if lang in ("en", "ko") else ""
    lines: list[str] = []
    cur: list[str] = []
    for unit in units:
        trial = sep.join([*cur, unit])
        if cur and font.getlength(trial) > max_width and unit[0] not in NO_LINE_START:
            lines.append(sep.join(cur))
            cur = [unit]
        else:
            cur.append(unit)
    if cur:
        lines.append(sep.join(cur))
    return lines


def render_text(
    lines: list[str],
    key: str,
    size: int,
    fill: tuple[int, int, int],
    vertical: bool = False,
    stroke: int = 0,
    stroke_fill: tuple[int, int, int] = (255, 255, 255),
) -> TextBlock:
    font = _font(key, size)
    pad = stroke + 2
    if vertical:
        cols = lines
        colw = int(size * 1.25)
        rows = max(len(c) for c in cols)
        w, h = colw * len(cols) + 2 * pad, int(size * 1.1) * rows + 2 * pad
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        for ci, col in enumerate(cols):  # first column on the right
            x = w - pad - (ci + 1) * colw + colw / 2
            for ri, ch in enumerate(col):
                y = pad + ri * int(size * 1.1)
                draw.text((x, y), ch, font=font, fill=(*fill, 255), anchor="mt",
                          stroke_width=stroke, stroke_fill=(*stroke_fill, 255))  # fmt: skip
    else:
        asc, desc = font.getmetrics()
        lh = int((asc + desc) * 1.15)
        w = int(max(font.getlength(ln) for ln in lines)) + 2 * pad
        h = lh * len(lines) + 2 * pad
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        for i, ln in enumerate(lines):
            draw.text((w / 2, pad + i * lh), ln, font=font, fill=(*fill, 255), anchor="ma",
                      stroke_width=stroke, stroke_fill=(*stroke_fill, 255))  # fmt: skip
    return TextBlock(np.asarray(img, dtype=np.uint8).copy(), lines)


def rotate_block(block: TextBlock, degrees: float) -> TextBlock:
    img = Image.fromarray(block.rgba, "RGBA").rotate(
        degrees, resample=Image.Resampling.BICUBIC, expand=True
    )
    return TextBlock(np.asarray(img, dtype=np.uint8).copy(), block.lines)


def paste(page: U8, block: TextBlock, x: int, y: int) -> Bool:
    """Alpha-composite ``block`` at (x, y); return the full-page text mask (alpha > 0)."""
    h, w = block.rgba.shape[:2]
    alpha = block.rgba[..., 3:4].astype(np.float32) / 255.0
    view = page[y : y + h, x : x + w].astype(np.float32)
    page[y : y + h, x : x + w] = np.rint(view * (1 - alpha) + block.rgba[..., :3] * alpha).astype(
        np.uint8
    )
    mask = np.zeros(page.shape[:2], bool)
    mask[y : y + h, x : x + w] = block.rgba[..., 3] > 0
    return mask


def erode(mask: Bool, radius: int) -> Bool:
    if radius <= 0:
        return mask.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    out = cv2.erode(mask.astype(np.uint8), k, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return np.asarray(out > 0, dtype=np.bool_)


def dilate(mask: Bool, radius: int) -> Bool:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return np.asarray(cv2.dilate(mask.astype(np.uint8), k) > 0, dtype=np.bool_)


def columns(text: str, per_column: int) -> list[str]:
    """Split vertical text into columns, never starting a column with closing punctuation."""
    cols: list[str] = []
    for ch in text:
        if cols and (len(cols[-1]) < per_column or ch in NO_LINE_START):
            cols[-1] += ch
        else:
            cols.append(ch)
    return cols
