"""Deterministic synthetic comic pages with exact ground truth.

Used by the test-suite, the benchmark and ``manga-arabic demo``. Everything is drawn
from code (no copyrighted material). Ground truth comes from the rendered ink itself:
text bounding boxes and masks are measured, never estimated.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont

from manga_ar.errors import ModelUnavailableError
from manga_ar.resources import fonts_dir
from manga_ar.schemas import BBox, CropMask

RgbArray = npt.NDArray[np.uint8]
BoolArray = npt.NDArray[np.bool_]
Shape = Literal["ellipse", "roundrect", "spiky", "rect"]

TEXTS: dict[str, list[str]] = {
    "ja": [
        "今日は晴れ",
        "本当に行くの",
        "待ってくれ",
        "ありがとう",
        "大丈夫だよ",
        "気をつけて",
        "どこへ行く",
        "もう遅いよ",
        "信じてるよ",
        "一緒に行こう",
    ],
    "ko": [
        "오늘은 맑아요",
        "정말 갈 거야",
        "기다려 줘",
        "고마워요",
        "괜찮아",
        "조심해",
        "어디 가니",
        "너무 늦었어",
    ],
    "zh": [
        "今天天气很好",
        "你真的要去吗",
        "等一下",
        "谢谢你",
        "没关系",
        "小心点",
        "你去哪里",
        "太晚了",
    ],
}

_FONT_FILES = {
    "ja": "NotoSansJP-Regular.otf",
    "ko": "NotoSansKR-Regular.otf",
    "zh": "NotoSansSC-Regular.otf",
}
_SYSTEM_CJK = [
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "C:/Windows/Fonts/msgothic.ttc",
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/msyh.ttc",
]


@dataclass
class GtRegion:
    """Ground truth for one text region."""

    id: str
    type: str
    lang: str
    text: str
    vertical: bool
    text_bbox: BBox
    text_mask: CropMask
    reading_order: int
    shape: str = "none"
    bubble_mask: CropMask | None = None
    fill: tuple[int, int, int] = (255, 255, 255)
    text_color: tuple[int, int, int] = (0, 0, 0)
    decorations: list[str] = field(default_factory=list)
    background: str = "uniform"  # uniform | gradient | texture

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "lang": self.lang,
            "text": self.text,
            "vertical": self.vertical,
            "text_bbox": self.text_bbox.to_list(),
            "reading_order": self.reading_order,
            "shape": self.shape,
            "bubble_bbox": None if self.bubble_mask is None else self.bubble_mask.bbox.to_list(),
            "fill": list(self.fill),
            "decorations": self.decorations,
            "background": self.background,
        }


@dataclass
class SynthPage:
    name: str
    image: RgbArray
    regions: list[GtRegion]
    lang: str
    reading_order: str
    tags: list[str] = field(default_factory=list)

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    def ground_truth(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "lang": self.lang,
            "reading_order": self.reading_order,
            "size": [self.width, self.height],
            "regions": [r.to_json() for r in sorted(self.regions, key=lambda r: r.reading_order)],
        }


# --------------------------------------------------------------------------- fonts
def find_cjk_font(lang: str, cache_dir: Path | None = None) -> Path:
    """Locate a CJK font: downloaded Noto subset first, then common system fonts."""
    candidates: list[Path] = []
    if cache_dir is not None:
        candidates.append(cache_dir / "models" / "fonts" / _FONT_FILES[lang])
    candidates.extend(Path(p) for p in _SYSTEM_CJK)
    if lang == "ja":
        candidates.append(Path("/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"))
    for path in candidates:
        if path.is_file():
            return path
    raise ModelUnavailableError(
        f"no CJK font for {lang!r}; run `manga-arabic models download fonts` "
        "(or scripts/download_assets.py)"
    )


def symbol_font_path(char: str = "\u2661") -> Path:
    """Symbols 2 covers ♡♥☆★; Noto Sans Symbols covers ♪♫."""
    name = (
        "NotoSansSymbols-Variable.ttf" if char in "\u266a\u266b" else "NotoSansSymbols2-Regular.ttf"
    )
    return fonts_dir() / name


class Fonts:
    """Cache of ImageFont objects keyed by (lang, size)."""

    def __init__(self, cache_dir: Path | None) -> None:
        self.cache_dir = cache_dir
        self._cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}

    def get(self, lang: str, size: int, char: str = "\u2661") -> ImageFont.FreeTypeFont:
        """``lang="sym"`` selects the symbol font covering ``char``."""
        key = (lang if lang != "sym" else f"sym:{symbol_font_path(char).name}", size)
        if key not in self._cache:
            path = symbol_font_path(char) if lang == "sym" else find_cjk_font(lang, self.cache_dir)
            self._cache[key] = ImageFont.truetype(str(path), size)
        return self._cache[key]


# ------------------------------------------------------------------------ builder
class PageBuilder:
    """Imperative page composer that records ground truth as it draws."""

    def __init__(
        self,
        width: int,
        height: int,
        *,
        seed: int,
        fonts: Fonts,
        lang: str,
        reading_order: str,
        name: str,
    ) -> None:
        self.rng = random.Random(seed)
        self.fonts = fonts
        self.lang = lang
        self.reading_order = reading_order
        self.name = name
        self.canvas = np.full((height, width, 3), 255, np.uint8)
        self.regions: list[GtRegion] = []
        self.occupied: list[BBox] = []
        self.tags: list[str] = []

    @property
    def height(self) -> int:
        return int(self.canvas.shape[0])

    @property
    def width(self) -> int:
        return int(self.canvas.shape[1])

    # ------------------------------------------------------------ artwork
    def panel(self, box: BBox, border: int = 4) -> None:
        cv2.rectangle(self.canvas, (box.x0, box.y0), (box.x1 - 1, box.y1 - 1), (0, 0, 0), border)

    def screentone(self, box: BBox, step: int = 6, dot: int = 2, value: int = 90) -> None:
        for y in range(box.y0, box.y1, step):
            for x in range(box.x0, box.x1, step):
                cv2.circle(self.canvas, (x, y), dot // 2, (value, value, value), -1)

    def gradient(self, box: BBox, c0: tuple[int, int, int], c1: tuple[int, int, int]) -> None:
        h, w = box.height, box.width
        t = np.linspace(0.0, 1.0, h)[:, None, None]
        grad = (np.array(c0) * (1 - t) + np.array(c1) * t).astype(np.uint8)
        self.canvas[box.y0 : box.y1, box.x0 : box.x1] = np.repeat(grad, w, axis=1)

    def hatching(self, box: BBox, step: int = 14) -> None:
        sub = np.ascontiguousarray(self.canvas[box.y0 : box.y1, box.x0 : box.x1])
        for k in range(-box.height, box.width, step):
            cv2.line(sub, (k, 0), (k + box.height, box.height), (60, 60, 60), 1)
        self.canvas[box.y0 : box.y1, box.x0 : box.x1] = sub

    # --------------------------------------------------------------- text
    def render_text(
        self,
        text: str,
        size: int,
        vertical: bool,
        lang: str | None = None,
        decorations: str = "",
        max_col: int = 4,
    ) -> Image.Image:
        """Render text to an L-mode ink image (255 = ink) tightly cropped."""
        lang = lang or self.lang
        font = self.fonts.get(lang, size)
        sym = self.fonts.get("sym", size, decorations[:1] or "\u2661")
        glyphs = list(text.replace(" ", "")) if vertical else [text]
        if vertical:
            cols = [glyphs[i : i + max_col] for i in range(0, len(glyphs), max_col)]
            step = int(size * 1.12)
            col_w = int(size * 1.45)
            w = len(cols) * col_w + size
            h = max(len(c) for c in cols) * step + size + (step if decorations else 0)
            img = Image.new("L", (w, h), 0)
            d = ImageDraw.Draw(img)
            for ci, col in enumerate(cols):
                x = w - size // 2 - (ci + 1) * col_w + (col_w - size) // 2
                for ri, ch in enumerate(col):
                    d.text((x, size // 2 + ri * step), ch, font=font, fill=255)
                if decorations and ci == len(cols) - 1:
                    d.text((x, size // 2 + len(col) * step), decorations, font=sym, fill=255)
        else:
            lines = text.split("\n")
            widths = [font.getlength(line) for line in lines]
            dec_w = sym.getlength(decorations) if decorations else 0.0
            w = int(max(widths) + dec_w + size)
            step = int(size * 1.3)
            h = len(lines) * step + size
            img = Image.new("L", (w, h), 0)
            d = ImageDraw.Draw(img)
            for li, line in enumerate(lines):
                xf = (w - widths[li] - (dec_w if li == len(lines) - 1 else 0)) / 2
                d.text((xf, size // 2 + li * step), line, font=font, fill=255)
                if decorations and li == len(lines) - 1:
                    d.text(
                        (xf + widths[li], size // 2 + li * step), decorations, font=sym, fill=255
                    )
        bbox = img.getbbox()
        return img.crop(bbox) if bbox else img

    def _stamp_text(
        self, ink: Image.Image, center: tuple[int, int], color: tuple[int, int, int]
    ) -> tuple[BBox, BoolArray]:
        arr = np.asarray(ink, dtype=np.uint8)
        h, w = arr.shape
        x0 = int(center[0] - w / 2)
        y0 = int(center[1] - h / 2)
        box = BBox(x0, y0, x0 + w, y0 + h)
        region = self.canvas[y0 : y0 + h, x0 : x0 + w].astype(np.float32)
        alpha = (arr.astype(np.float32) / 255.0)[..., None]
        blended = region * (1 - alpha) + np.array(color, np.float32) * alpha
        self.canvas[y0 : y0 + h, x0 : x0 + w] = np.rint(blended).astype(np.uint8)
        full = np.zeros((self.height, self.width), np.bool_)
        full[y0 : y0 + h, x0 : x0 + w] = arr > 96
        tight = BBox.from_mask(full) or box
        return tight, full

    # ------------------------------------------------------------- shapes
    def shape_mask(self, shape: Shape, box: BBox, tail: tuple[int, int] | None = None) -> BoolArray:
        img = Image.new("L", (self.width, self.height), 0)
        d = ImageDraw.Draw(img)
        rect = (box.x0, box.y0, box.x1 - 1, box.y1 - 1)
        cx, cy = box.center
        if shape == "ellipse":
            d.ellipse(rect, fill=255)
        elif shape == "roundrect":
            d.rounded_rectangle(rect, radius=min(box.width, box.height) // 4, fill=255)
        elif shape == "rect":
            d.rectangle(rect, fill=255)
        else:  # spiky: star polygon around the ellipse
            spikes = 18
            pts = []
            for k in range(spikes * 2):
                ang = math.pi * k / spikes
                r = 1.0 if k % 2 == 0 else 0.8
                pts.append(
                    (
                        cx + r * box.width / 2 * math.cos(ang),
                        cy + r * box.height / 2 * math.sin(ang),
                    )
                )
            d.polygon(pts, fill=255)
        if tail is not None:
            tx, ty = tail
            base = 0.18 * min(box.width, box.height)
            ang = math.atan2(ty - cy, tx - cx)
            ex = cx + 0.42 * box.width * math.cos(ang)
            ey = cy + 0.42 * box.height * math.sin(ang)
            px, py = -math.sin(ang) * base, math.cos(ang) * base
            d.polygon([(ex + px, ey + py), (ex - px, ey - py), (tx, ty)], fill=255)
        return np.asarray(img) > 0

    def draw_bubble(
        self,
        mask: BoolArray,
        fill: tuple[int, int, int] | str,
        outline: int = 3,
        outline_color: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        m8 = mask.astype(np.uint8)
        if isinstance(fill, str):  # "gradient"
            box = BBox.from_mask(mask)
            assert box is not None
            t = np.linspace(0, 1, box.height)[:, None, None]
            grad = (np.array((255, 236, 210)) * (1 - t) + np.array((205, 225, 255)) * t).astype(
                np.uint8
            )
            layer = np.repeat(grad, box.width, axis=1)
            sub = mask[box.y0 : box.y1, box.x0 : box.x1]
            self.canvas[box.y0 : box.y1, box.x0 : box.x1][sub] = layer[sub]
        else:
            self.canvas[mask] = fill
        if outline > 0:
            eroded = cv2.erode(m8, np.ones((2 * outline + 1, 2 * outline + 1), np.uint8))
            ring = (m8 > 0) & (eroded == 0)
            self.canvas[ring] = outline_color

    # ------------------------------------------------------------ regions
    def add_region(
        self,
        kind: str,
        text: str,
        box_hint: BBox,
        *,
        shape: Shape = "ellipse",
        vertical: bool | None = None,
        size: int = 28,
        fill: tuple[int, int, int] | str = (255, 255, 255),
        text_color: tuple[int, int, int] = (0, 0, 0),
        tail: tuple[int, int] | None = None,
        decorations: str = "",
        lang: str | None = None,
        background: str = "uniform",
        furigana: bool = False,
        outline: int = 3,
    ) -> GtRegion:
        """Draw a region whose container is ``box_hint`` and record ground truth."""
        lang = lang or self.lang
        vertical = (lang == "ja") if vertical is None else vertical
        ink = self.render_text(text, size, vertical, lang, decorations)
        center = (int(box_hint.center[0]), int(box_hint.center[1]))
        bubble_mask: CropMask | None = None
        if kind in {"bubble", "narration"}:
            mask = self.shape_mask("rect" if kind == "narration" else shape, box_hint, tail)
            width = outline if kind == "bubble" else 2
            self.draw_bubble(mask, fill, outline=width)
            # Ground truth is the fillable interior (outline excluded), which is what the
            # segmenter, the inpainter's allowed zone and the typesetter work with.
            interior = cv2.erode(mask.astype(np.uint8), np.ones((2 * width + 1,) * 2, np.uint8))
            bubble_mask = CropMask.from_full(interior > 0)
        text_box, text_full = self._stamp_text(ink, center, text_color)
        if furigana and vertical:
            small = self.render_text("ふりがな", max(8, size // 2), True, "ja", max_col=4)
            fx = text_box.x1 + 3 + small.width // 2
            fy = text_box.y0 + small.height // 2
            self._stamp_text(small, (fx, fy), text_color)
            self.tags.append("furigana")
        region = GtRegion(
            id=f"r{len(self.regions)}",
            type=kind,
            lang=lang,
            text=text,
            vertical=vertical,
            text_bbox=text_box,
            text_mask=CropMask.from_full(text_full) or CropMask(text_box, np.zeros((1, 1), bool)),
            reading_order=len(self.regions),
            shape=shape if kind == "bubble" else ("rect" if kind == "narration" else "none"),
            bubble_mask=bubble_mask,
            fill=fill if isinstance(fill, tuple) else (230, 230, 232),
            text_color=text_color,
            decorations=list(decorations),
            background="gradient" if fill == "gradient" else background,
        )
        self.regions.append(region)
        self.occupied.append(box_hint)
        return region

    def bubble_box_for(self, text: str, size: int, vertical: bool, scale: float = 1.6) -> BBox:
        ink = self.render_text(text, size, vertical)
        w = int(ink.width * scale + size)
        h = int(ink.height * scale + size)
        return BBox(0, 0, w, h)

    def place(self, size_box: BBox, area: BBox, margin: int = 12) -> BBox | None:
        """Random non-overlapping placement of a box of ``size_box`` dims inside ``area``."""
        w, h = size_box.width, size_box.height
        if w > area.width - 2 * margin or h > area.height - 2 * margin:
            return None
        for _ in range(200):
            x0 = self.rng.randint(area.x0 + margin, area.x1 - margin - w)
            y0 = self.rng.randint(area.y0 + margin, area.y1 - margin - h)
            cand = BBox(x0, y0, x0 + w, y0 + h)
            if all(cand.expand(margin).intersection(o) is None for o in self.occupied):
                return cand
        return None

    def build(self) -> SynthPage:
        return SynthPage(
            name=self.name,
            image=self.canvas,
            regions=self.regions,
            lang=self.lang,
            reading_order=self.reading_order,
            tags=sorted(set(self.tags)),
        )


def _default_order(lang: str) -> str:
    return {"ja": "manga_rtl", "zh": "comic_ltr", "ko": "webtoon_ttb"}[lang]


def _panels(width: int, height: int, rows: int, cols: int, gutter: int = 20) -> list[list[BBox]]:
    grid = []
    ph = (height - gutter * (rows + 1)) // rows
    pw = (width - gutter * (cols + 1)) // cols
    for r in range(rows):
        row = []
        for c in range(cols):
            x0 = gutter + c * (pw + gutter)
            y0 = gutter + r * (ph + gutter)
            row.append(BBox(x0, y0, x0 + pw, y0 + ph))
        grid.append(row)
    return grid


def _ordered_panels(grid: list[list[BBox]], order: str) -> list[BBox]:
    out: list[BBox] = []
    for row in grid:
        out.extend(reversed(row) if order == "manga_rtl" else row)
    return out


# ---------------------------------------------------------------------- scenarios
def basic_page(
    lang: str = "ja",
    seed: int = 0,
    cache_dir: Path | None = None,
    width: int = 900,
    vertical: bool | None = None,
) -> SynthPage:
    """2×2 panels with one or two plain bubbles each (ellipse/roundrect, tails).

    ``vertical=False`` renders Japanese horizontally (read comic_ltr)."""
    height = int(width * 1.4)
    is_vertical = (lang == "ja") if vertical is None else vertical
    order = (
        _default_order(lang)
        if is_vertical == (lang == "ja")
        else ("manga_rtl" if is_vertical else "comic_ltr")
    )
    b = PageBuilder(
        width,
        height,
        seed=seed,
        fonts=Fonts(cache_dir),
        lang=lang,
        reading_order=order,
        name=f"basic_{lang}{'' if vertical is None else ('_v' if is_vertical else '_h')}_{seed}",
    )
    grid = _panels(width, height, 2, 2)
    texts = list(TEXTS[lang])
    b.rng.shuffle(texts)
    for panel in _ordered_panels(grid, order):
        b.panel(panel)
    for panel in _ordered_panels(grid, order):
        n = 2 if b.rng.random() < 0.6 else 1
        halves = _split_panel(panel, n, order)
        for area in halves:
            text = texts.pop()
            size = b.rng.choice([26, 28, 32])
            dims = b.bubble_box_for(text, size, is_vertical, scale=1.6 if is_vertical else 1.35)
            spot = b.place(dims, area)
            if spot is None:
                continue
            shape: Shape = b.rng.choice(["ellipse", "roundrect"])
            tail = (int(spot.center[0]) + b.rng.randint(-20, 20), min(spot.y1 + 30, area.y1 - 6))
            b.add_region(
                "bubble", text, spot, shape=shape, vertical=is_vertical, size=size, tail=tail
            )
    return b.build()


def _split_panel(panel: BBox, n: int, order: str) -> list[BBox]:
    """Split a panel into ``n`` reading-ordered sub-areas (diagonal to keep order clear)."""
    if n == 1:
        return [panel]
    if order != "manga_rtl":  # horizontal text: stacked halves, read top → bottom
        midy = (panel.y0 + panel.y1) // 2
        return [BBox(panel.x0, panel.y0, panel.x1, midy), BBox(panel.x0, midy, panel.x1, panel.y1)]
    midx = (panel.x0 + panel.x1) // 2
    midy = (panel.y0 + panel.y1) // 2
    right_top = BBox(midx, panel.y0, panel.x1, midy + (panel.y1 - midy) // 3)
    left_bottom = BBox(panel.x0, midy - (midy - panel.y0) // 3, midx, panel.y1)
    left_top = BBox(panel.x0, panel.y0, midx, midy + (panel.y1 - midy) // 3)
    right_bottom = BBox(midx, midy - (midy - panel.y0) // 3, panel.x1, panel.y1)
    if order == "manga_rtl":
        return [right_top, left_bottom]
    return [left_top, right_bottom]


def variety_page(lang: str = "ja", seed: int = 0, cache_dir: Path | None = None) -> SynthPage:
    """Spiky, dark, tinted and gradient bubbles, a narration box, free text over a
    gradient, screentone/hatching artwork and an SFX."""
    width, height = 1000, 1400
    order = _default_order(lang)
    b = PageBuilder(
        width,
        height,
        seed=seed,
        fonts=Fonts(cache_dir),
        lang=lang,
        reading_order=order,
        name=f"variety_{lang}_{seed}",
    )
    grid = _panels(width, height, 3, 2)
    texts = list(TEXTS[lang])
    b.rng.shuffle(texts)
    vertical = lang == "ja"
    panels = _ordered_panels(grid, order)
    # artwork first so bubbles sit on top
    b.screentone(BBox(panels[1].x0 + 10, panels[1].y0 + 10, panels[1].x1 - 10, panels[1].y1 - 10))
    b.gradient(panels[4], (120, 150, 210), (240, 200, 170))
    b.hatching(BBox(panels[5].x0 + 5, panels[5].y0 + 5, panels[5].x1 - 5, panels[5].y1 - 5))
    for p in panels:
        b.panel(p)
    specs: list[dict[str, Any]] = [
        {"kind": "bubble", "shape": "spiky", "fill": (255, 255, 255)},
        {"kind": "bubble", "shape": "ellipse", "fill": (20, 20, 20), "text_color": (255, 255, 255)},
        {"kind": "bubble", "shape": "roundrect", "fill": (255, 238, 205)},
        {"kind": "bubble", "shape": "ellipse", "fill": "gradient"},
        {"kind": "free_text", "text_color": (20, 20, 20), "background": "gradient"},
        {"kind": "narration", "fill": (255, 255, 255)},
    ]
    for panel, spec in zip(panels, specs, strict=True):
        text = texts.pop()
        size = 28
        kind = spec["kind"]
        v = vertical and kind != "narration"
        dims = b.bubble_box_for(text, size, v, scale=1.2 if kind == "free_text" else 1.7)
        spot = b.place(dims, panel, margin=18)
        if spot is None:
            continue
        b.add_region(
            kind,
            text,
            spot,
            shape=spec.get("shape", "ellipse"),
            vertical=v,
            size=size,
            fill=spec.get("fill", (255, 255, 255)),
            text_color=spec.get("text_color", (0, 0, 0)),
            background=spec.get("background", "uniform"),
        )
    # A large SFX in the hatched panel (not recorded as translatable text by default).
    sfx_font = b.fonts.get(lang, 90)
    img = Image.fromarray(b.canvas)
    d = ImageDraw.Draw(img)
    p = panels[5]
    sfx = {"ja": "ドン", "ko": "쾅", "zh": "砰"}[lang]
    left, top, right, bottom = (
        int(v) for v in d.textbbox((0, 0), sfx, font=sfx_font, stroke_width=4)
    )
    corners = [
        (p.x0 + 20, p.y1 - 20 - bottom),
        (p.x1 - 20 - right, p.y1 - 20 - bottom),
        (p.x0 + 20, p.y0 + 20),
        (p.x1 - 20 - right, p.y0 + 20),
    ]
    for x, y in corners:
        box = BBox(x + left, y + top, x + right, y + bottom)
        if all(box.expand(8).intersection(o) is None for o in b.occupied):
            d.text(
                (x, y),
                sfx,
                font=sfx_font,
                fill=(0, 0, 0),
                stroke_width=4,
                stroke_fill=(255, 255, 255),
            )
            b.occupied.append(box)
            break
    b.canvas = np.asarray(img).copy()
    b.tags.append("sfx")
    return b.build()


def furigana_page(seed: int = 0, cache_dir: Path | None = None) -> SynthPage:
    width, height = 800, 1000
    b = PageBuilder(
        width,
        height,
        seed=seed,
        fonts=Fonts(cache_dir),
        lang="ja",
        reading_order="manga_rtl",
        name=f"furigana_{seed}",
    )
    grid = _panels(width, height, 1, 2)
    panels = _ordered_panels(grid, "manga_rtl")
    for p in panels:
        b.panel(p)
    for panel, text in zip(panels, ["信じてるよ", "一緒に行こう"], strict=True):
        dims = b.bubble_box_for(text, 30, True, scale=1.9)
        spot = b.place(dims, panel)
        if spot is not None:
            b.add_region("bubble", text, spot, vertical=True, size=30, furigana=True)
    return b.build()


def symbols_page(lang: str = "ja", seed: int = 0, cache_dir: Path | None = None) -> SynthPage:
    width, height = 800, 600
    b = PageBuilder(
        width,
        height,
        seed=seed,
        fonts=Fonts(cache_dir),
        lang=lang,
        reading_order=_default_order(lang),
        name=f"symbols_{lang}_{seed}",
    )
    b.panel(BBox(20, 20, width - 20, height - 20))
    areas = [BBox(width // 2, 20, width - 20, height - 20), BBox(20, 20, width // 2, height - 20)]
    if lang != "ja":
        areas.reverse()
    for area, (text, dec) in zip(
        areas, [(TEXTS[lang][3], "♡"), (TEXTS[lang][4], "♪")], strict=True
    ):
        vertical = lang == "ja"
        dims = b.bubble_box_for(text + dec, 30, vertical, scale=1.8)
        spot = b.place(dims, area)
        if spot is not None:
            b.add_region("bubble", text, spot, vertical=vertical, size=30, decorations=dec)
    return b.build()


def webtoon_strip(
    seed: int = 0, cache_dir: Path | None = None, lang: str = "ko", seams: tuple[int, ...] = ()
) -> SynthPage:
    """Tall strip (720×6400) with bubbles centred on likely tile seams and in between."""
    width, height = 720, 6400
    b = PageBuilder(
        width,
        height,
        seed=seed,
        fonts=Fonts(cache_dir),
        lang=lang,
        reading_order="webtoon_ttb",
        name=f"webtoon_{lang}_{seed}",
    )
    # Artwork blocks separated by white gutters, as in real webtoons.
    for y in range(100, height - 200, 900):
        b.gradient(BBox(40, y, width - 40, y + 600), (200, 210, 230), (235, 220, 200))
    centers = list(seams) or [1900, 2050, 3500, 3900, 5400, 700]
    texts = list(TEXTS[lang])
    b.rng.shuffle(texts)
    for cy in sorted(centers):
        text = texts.pop() if texts else TEXTS[lang][0]
        dims = b.bubble_box_for(text, 30, lang == "ja", scale=1.8)
        cx = b.rng.randint(dims.width // 2 + 30, width - dims.width // 2 - 30)
        spot = BBox(
            cx - dims.width // 2, cy - dims.height // 2, cx + dims.width // 2, cy + dims.height // 2
        )
        if any(spot.expand(10).intersection(o) for o in b.occupied):
            continue
        b.add_region("bubble", text, spot, vertical=lang == "ja", size=30)
    return b.build()


def adversarial_gutter_page(seed: int = 0, cache_dir: Path | None = None) -> SynthPage:
    """A bubble whose outline is open where it touches the panel border, so its white
    interior connects with the gutter: naive region growing leaks into the page."""
    width, height = 800, 900
    b = PageBuilder(
        width,
        height,
        seed=seed,
        fonts=Fonts(cache_dir),
        lang="ja",
        reading_order="manga_rtl",
        name=f"adversarial_{seed}",
    )
    panel = BBox(20, 20, width - 20, height - 20)
    b.panel(panel)
    text = "どこへ行く"
    dims = b.bubble_box_for(text, 30, True, scale=1.7)
    # Straddle the top panel border so the bubble merges with the top gutter.
    x0 = width // 2 - dims.width // 2
    spot = BBox(x0, 0, x0 + dims.width, dims.height)
    region = b.add_region("bubble", text, spot, vertical=True, size=30)
    # Erase the panel border and the bubble outline where they meet the gutter.
    b.canvas[0:30, x0 + 10 : x0 + dims.width - 10] = np.where(
        b.canvas[0:30, x0 + 10 : x0 + dims.width - 10] < 128,
        255,
        b.canvas[0:30, x0 + 10 : x0 + dims.width - 10],
    )
    b.tags.append("leak")
    region.shape = "open"
    return b.build()


def texture_page(seed: int = 0, cache_dir: Path | None = None, lang: str = "ja") -> SynthPage:
    """Free text directly over screentone and over hatching (exercises LaMa / Telea)."""
    width, height = 900, 700
    b = PageBuilder(
        width,
        height,
        seed=seed,
        fonts=Fonts(cache_dir),
        lang=lang,
        reading_order=_default_order(lang),
        name=f"texture_{lang}_{seed}",
    )
    left, right = BBox(20, 20, 440, 680), BBox(460, 20, 880, 680)
    b.screentone(BBox(left.x0 + 6, left.y0 + 6, left.x1 - 6, left.y1 - 6), step=5, dot=2, value=150)
    b.hatching(BBox(right.x0 + 6, right.y0 + 6, right.x1 - 6, right.y1 - 6), step=9)
    b.panel(left)
    b.panel(right)
    texts = list(TEXTS[lang])
    b.rng.shuffle(texts)
    areas = [right, left] if lang == "ja" else [left, right]
    for k, area in enumerate(areas):
        text = texts.pop()
        dims = b.bubble_box_for(text, 34, lang == "ja", scale=1.1)
        spot = b.place(dims, area, margin=40)
        if spot is None:
            continue
        if k == 0:  # manga-style white halo behind free text on the first panel
            halo = np.asarray(b.render_text(text, 34, lang == "ja"))
            halo = cv2.dilate(halo, np.ones((9, 9), np.uint8))
            center = (int(spot.center[0]), int(spot.center[1]))
            b._stamp_text(Image.fromarray(halo), center, (255, 255, 255))
        b.add_region("free_text", text, spot, vertical=lang == "ja", size=34, background="texture")
    b.tags.append("texture")
    return b.build()


def two_block_bubble_page(seed: int = 0, cache_dir: Path | None = None) -> SynthPage:
    """One large bubble holding two separated text blocks (must become one MERGED region)."""
    width, height = 700, 700
    b = PageBuilder(
        width,
        height,
        seed=seed,
        fonts=Fonts(cache_dir),
        lang="ja",
        reading_order="manga_rtl",
        name=f"two_block_{seed}",
    )
    b.panel(BBox(20, 20, width - 20, height - 20))
    box = BBox(120, 120, 580, 580)
    mask = b.shape_mask("ellipse", box)
    b.draw_bubble(mask, (255, 255, 255))
    ink_a = b.render_text("待ってくれ", 30, True)
    ink_b = b.render_text("行こう", 30, True)
    ta, fa = b._stamp_text(ink_a, (420, 330), (0, 0, 0))
    tb, fb = b._stamp_text(ink_b, (270, 360), (0, 0, 0))
    interior = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    full = fa | fb
    b.regions.append(
        GtRegion(
            id="r0",
            type="bubble",
            lang="ja",
            text="待ってくれ行こう",
            vertical=True,
            text_bbox=ta.union(tb),
            text_mask=CropMask.from_full(full) or CropMask(ta, np.zeros((1, 1), bool)),
            reading_order=0,
            shape="ellipse",
            bubble_mask=CropMask.from_full(interior),
        )
    )
    b.tags.append("merge")
    return b.build()


DEMO_TEXTS = ["你真的要去吗", "等一下", "谢谢你"]
DEMO_TRANSLATIONS = {
    "你真的要去吗": "هل ستذهب حقًا؟",
    "等一下": "انتظر لحظة!",
    "谢谢你": "شكرًا لك",
}


def demo_page(cache_dir: Path | None = None) -> SynthPage:
    """Fixed Chinese page used by ``manga-arabic demo`` (paired with DEMO_TRANSLATIONS)."""
    width, height = 900, 700
    b = PageBuilder(
        width,
        height,
        seed=7,
        fonts=Fonts(cache_dir),
        lang="zh",
        reading_order="comic_ltr",
        name="demo_zh",
    )
    left, right = BBox(20, 20, 440, 680), BBox(460, 20, 880, 680)
    b.panel(left)
    b.panel(right)
    b.gradient(BBox(470, 380, 870, 670), (215, 225, 240), (245, 225, 205))
    b.panel(right)
    b.add_region(
        "bubble",
        DEMO_TEXTS[0],
        BBox(60, 60, 400, 300),
        shape="ellipse",
        vertical=False,
        size=34,
        tail=(220, 380),
    )
    # Regions are added in reading order (comic_ltr: left panel top→bottom, then right).
    b.add_region("narration", DEMO_TEXTS[2], BBox(100, 520, 360, 620), vertical=False, size=30)
    b.add_region(
        "bubble",
        DEMO_TEXTS[1],
        BBox(520, 80, 820, 280),
        shape="roundrect",
        vertical=False,
        size=34,
        tail=(640, 360),
    )
    return b.build()
