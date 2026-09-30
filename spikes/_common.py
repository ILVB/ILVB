"""Shared helpers for spikes: synthetic CJK text images (no copyrighted material)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "spikes" / "out"
CACHE = Path(os.environ.get("MANGAAR_CACHE_DIR", ROOT / ".cache"))
FONT_FILES = {
    "ja": CACHE / "fonts/NotoSansJP-Regular.otf",
    "ko": CACHE / "fonts/NotoSansKR-Regular.otf",
    "zh": CACHE / "fonts/NotoSansSC-Regular.otf",
}


def text_image(text: str, lang: str, size: int = 40, vertical: bool = False) -> np.ndarray:
    """Render black text on white. Vertical = one glyph per row, columns right→left."""
    font = ImageFont.truetype(str(FONT_FILES[lang]), size)
    pad = size // 2
    if not vertical:
        l, t, r, b = font.getbbox(text)
        img = Image.new("RGB", (r - l + 2 * pad, b - t + 2 * pad), "white")
        ImageDraw.Draw(img).text((pad - l, pad - t), text, font=font, fill="black")
        return np.asarray(img)
    columns = text.split("\n")
    step = int(size * 1.1)
    height = max(len(c) for c in columns) * step + 2 * pad
    width = len(columns) * int(size * 1.4) + 2 * pad
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    for ci, col in enumerate(columns):
        x = width - pad - (ci + 1) * int(size * 1.4) + int(size * 0.2)
        for ri, ch in enumerate(col):
            draw.text((x, pad + ri * step), ch, font=font, fill="black")
    return np.asarray(img)
