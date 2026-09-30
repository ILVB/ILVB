"""Compose one synthetic page for a category, with exact ground truth."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from benchmarks import rle
from benchmarks.generators import GENERATOR_VERSION
from benchmarks.generators.draw import (
    U8,
    Bool,
    TextBlock,
    artwork,
    body_mask,
    columns,
    dilate,
    erode,
    gradient,
    paste,
    render_text,
    rotate_block,
    screentone,
    tail_polygon,
    wrap,
)
from benchmarks.generators.fonts import DISPLAY_FONT, TEXT_FONT
from benchmarks.generators.texts import TextPools

W, H = 1000, 1400
CATEGORIES = (
    "flat_white", "screentone", "dark_bubble", "text_on_art", "vertical_ja",
    "stylized", "sfx", "low_contrast", "tiny_text", "tails_overlap",
)  # fmt: skip
TEXTURED = {"screentone", "text_on_art", "sfx"}


@dataclass
class Placed:
    type: str
    text: str
    source_id: str
    lang: str
    vertical: bool
    text_mask: Bool
    bubble: Bool | None
    safe: Bool
    font: str
    size: int
    background: str
    panel: int
    x: int
    y: int
    extras: dict[str, Any] = field(default_factory=dict)


def _panels() -> list[tuple[int, int, int, int]]:
    m, g, rows, cols = 30, 18, 3, 2
    pw, ph = (W - 2 * m - (cols - 1) * g) // cols, (H - 2 * m - (rows - 1) * g) // rows
    return [(m + c * (pw + g), m + r * (ph + g), m + c * (pw + g) + pw, m + r * (ph + g) + ph)
            for r in range(rows) for c in range(cols)]  # fmt: skip


def _panel_order(direction: str) -> list[int]:
    order = []
    for r in range(3):
        cols = [1, 0] if direction == "rtl" else [0, 1]
        order += [r * 2 + c for c in cols]
    return order


class PageBuilder:
    def __init__(self, page_id: str, category: str, lang: str, seed: int, pools: TextPools) -> None:
        self.page_id, self.category, self.lang, self.seed = page_id, category, lang, seed
        self.rng = np.random.default_rng(seed)
        self.pools = pools
        self.page = np.full((H, W, 3), 255, np.uint8)
        self.clean = self.page.copy()
        self.occupied = np.zeros((H, W), bool)
        self.text_occupied = np.zeros((H, W), bool)
        self.placed: list[Placed] = []

    # -------------------------------------------------------------- panels
    def _panel_art(self, box: tuple[int, int, int, int]) -> None:
        x0, y0, x1, y1 = box
        if self.category in TEXTURED:
            art = artwork(y1 - y0, x1 - x0, self.rng)
        elif self.rng.random() < 0.3:
            art = gradient(y1 - y0, x1 - x0, 255, int(self.rng.integers(215, 245)))
        else:
            return
        for img in (self.page, self.clean):
            img[y0:y1, x0:x1] = art

    def _frame(self, box: tuple[int, int, int, int]) -> None:
        x0, y0, x1, y1 = box
        for img in (self.page, self.clean):
            cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 0), 3)

    # -------------------------------------------------------------- styles
    def _style(self) -> dict[str, Any]:
        c, rng = self.category, self.rng
        style: dict[str, Any] = {
            "font": TEXT_FONT[self.lang],
            "size": int(rng.integers(22, 31)),
            "fill": (int(rng.integers(0, 30)),) * 3,
            "stroke": 0,
            "bubble_fill": "white",
            "shape": str(rng.choice(["ellipse", "roundrect"])),
            "vertical": False,
            "type": "dialogue",
        }
        if self.lang == "en":
            style["size"] = int(rng.integers(20, 27))
        if c == "screentone":
            style["bubble_fill"] = "tone"
        elif c == "dark_bubble":
            style["bubble_fill"] = "dark"
            style["fill"] = (int(rng.integers(235, 256)),) * 3
        elif c == "vertical_ja":
            style["vertical"] = True
            style["shape"] = "ellipse"
        elif c == "stylized":
            if rng.random() < 0.5:
                style["font"] = DISPLAY_FONT[self.lang]
            else:
                style["fill"], style["stroke"] = (255, 255, 255), int(rng.integers(2, 4))
            style["size"] = int(rng.integers(26, 35))
            style["shape"] = str(rng.choice(["ellipse", "spiky"]))
        elif c == "low_contrast":
            base = int(rng.integers(200, 231))
            style["bubble_fill"] = ("gray", base)
            style["fill"] = (base - int(rng.integers(45, 71)),) * 3
        elif c == "tiny_text":
            style["size"] = int(rng.integers(10, 14))
        if c == "flat_white" and rng.random() < 0.25:
            style["shape"], style["type"] = "rect", "narration"
        return style

    def _fill(self, h: int, w: int, how: Any) -> U8:
        if how == "white":
            return np.full((h, w, 3), 255, np.uint8)
        if how == "dark":
            return np.full((h, w, 3), int(self.rng.integers(15, 46)), np.uint8)
        if how == "tone":
            if self.rng.random() < 0.6:
                return screentone(h, w, self.rng, bg=int(self.rng.integers(235, 251)),
                                  fg=int(self.rng.integers(150, 200)))  # fmt: skip
            return gradient(
                h, w, int(self.rng.integers(205, 250)), int(self.rng.integers(205, 250))
            )
        return np.full((h, w, 3), int(how[1]), np.uint8)

    # -------------------------------------------------------------- regions
    def _text_block(self, style: dict[str, Any], text: str, max_w: int) -> TextBlock:
        if style["vertical"]:
            clean = text.replace("…", "")
            cols = columns(clean, int(self.rng.integers(4, 7)))
            return render_text(
                cols, style["font"], style["size"], style["fill"], True, style["stroke"]
            )
        lines = wrap(text, style["font"], style["size"], max_w, self.lang)
        return render_text(lines, style["font"], style["size"], style["fill"], False,
                           style["stroke"], (0, 0, 0))  # fmt: skip

    def bubble_region(
        self, panel: int, box: tuple[int, int, int, int], shift: tuple[int, int] = (0, 0)
    ) -> bool:
        style = self._style()
        source_id, text = self.pools.dialogue(self.rng, self.lang)
        x0, y0, x1, y1 = box
        pw, ph = x1 - x0, y1 - y0
        block = self._text_block(style, text, int(pw * 0.45))
        bh_, bw_ = block.rgba.shape[:2]
        k = 1.12 if style["shape"] == "rect" else (1.5 if style["shape"] == "spiky" else 1.45)
        bw, bh = int(bw_ * k + 24), int(bh_ * k + 24)
        if bw > pw - 20 or bh > ph - 60:
            return False
        bx = int(self.rng.integers(x0 + 10, x1 - bw - 10 + 1)) + shift[0]
        by = int(self.rng.integers(y0 + 10, y1 - bh - 50 + 1)) + shift[1]
        bx, by = min(max(bx, x0 + 6), x1 - bw - 6), min(max(by, y0 + 6), y1 - bh - 6)
        body = np.zeros((H, W), bool)
        body[by : by + bh, bx : bx + bw] = body_mask(style["shape"], bw, bh, self.rng)
        allow_overlap = self.category == "tails_overlap"
        if (self.occupied & dilate(body, 4)).any() and not allow_overlap:
            return False
        if (self.text_occupied & dilate(body, 30)).any():
            return False  # never hide (or crowd) earlier lettering
        fill = self._fill(bh, bw, style["bubble_fill"])
        with_tail = style["shape"] != "rect" and (allow_overlap or self.rng.random() < 0.5)
        tail = np.zeros((H, W), np.uint8)
        if with_tail:
            cv2.fillPoly(tail, [np.array(tail_polygon((bx, by, bx + bw, by + bh), self.rng))], 1)
        shape = body | (tail > 0)
        outline = dilate(shape, 2) & ~shape
        for img in (self.page, self.clean):
            img[outline] = (0, 0, 0)
            region_fill = np.zeros_like(img)
            region_fill[by : by + bh, bx : bx + bw] = fill
            region_fill[tail > 0] = fill.reshape(-1, 3)[0]
            img[shape] = region_fill[shape]
        # an overlapping later bubble hides part of earlier ones
        for p in self.placed:
            if p.bubble is not None:
                p.bubble &= ~dilate(shape, 2)
                p.safe &= ~dilate(shape, 4)
        tx = bx + (bw - bw_) // 2
        ty = by + (bh - bh_) // 2
        mask = paste(self.page, block, tx, ty)
        margin = max(3, round(0.06 * min(bw, bh)))
        self.occupied |= dilate(shape, 2)
        self.text_occupied |= mask
        self.placed.append(Placed(
            type=style["type"], text=text if not style["vertical"] else text.replace("…", ""),
            source_id=source_id, lang=self.lang, vertical=style["vertical"], text_mask=mask,
            bubble=body.copy(), safe=erode(body, margin), font=style["font"], size=style["size"],
            background="textured" if style["bubble_fill"] == "tone" else "flat",
            panel=panel, x=tx, y=ty,
        ))  # fmt: skip
        return True

    def art_region(self, panel: int, box: tuple[int, int, int, int], sfx: bool) -> bool:
        rng = self.rng
        x0, y0, x1, y1 = box
        if sfx:
            source_id, text = self.pools.sfx(rng, self.lang)
            fill = (255, 255, 255) if rng.random() < 0.5 else (20, 20, 20)
            stroke_fill = (0, 0, 0) if fill[0] > 128 else (255, 255, 255)
            size = int(rng.integers(60, 101))
            block = render_text([text], DISPLAY_FONT[self.lang], size, fill, False, 4, stroke_fill)
            block = rotate_block(block, float(rng.uniform(-25, 25)))
            rtype, font = "sfx", DISPLAY_FONT[self.lang]
        else:
            source_id, text = self.pools.dialogue(rng, self.lang)
            size = int(rng.integers(22, 29))
            font = TEXT_FONT[self.lang]
            lines = wrap(text, font, size, int((x1 - x0) * 0.6), self.lang)
            dark = rng.random() < 0.5
            block = render_text(lines, font, size, (15, 15, 15) if dark else (255, 255, 255), False,
                                3, (255, 255, 255) if dark else (0, 0, 0))  # fmt: skip
            rtype = str(rng.choice(["narration", "sign"]))
        bh, bw = block.rgba.shape[:2]
        if bw > (x1 - x0) - 20 or bh > (y1 - y0) - 20:
            return False
        tx = int(rng.integers(x0 + 10, x1 - bw - 10 + 1))
        ty = int(rng.integers(y0 + 10, y1 - bh - 10 + 1))
        probe = np.zeros((H, W), bool)
        probe[ty : ty + bh, tx : tx + bw] = True
        if (self.occupied & dilate(probe, 6)).any():
            return False
        mask = paste(self.page, block, tx, ty)
        self.text_occupied |= mask
        hull = cv2.convexHull(np.column_stack(np.nonzero(mask)[::-1]).astype(np.int32))
        hull_mask = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(hull_mask, hull, 1)
        self.occupied |= dilate(probe, 6)
        self.placed.append(Placed(
            type=rtype, text=text, source_id=source_id, lang=self.lang, vertical=False,
            text_mask=mask, bubble=None, safe=dilate(hull_mask > 0, max(3, round(0.3 * size))),
            font=font, size=size, background="textured", panel=panel, x=tx, y=ty,
        ))  # fmt: skip
        return True


def make_page(
    page_id: str, category: str, lang: str, split: str, seed: int, pools: TextPools,
    refs: dict[str, dict[str, list[str]]] | None = None,
) -> tuple[U8, U8, dict[str, Any]]:  # fmt: skip
    """Render one page. Returns (image, clean background, GtPage fields without paths)."""
    b = PageBuilder(page_id, category, lang, seed, pools)
    direction = "rtl" if lang == "ja" else "ltr"
    boxes = _panels()
    for box in boxes:
        b._panel_art(box)
        b._frame(box)
    for panel, box in enumerate(boxes):
        if len(b.placed) >= 3 and b.rng.random() < 0.15:
            continue
        for _attempt in range(10):
            if category == "text_on_art":
                ok = b.art_region(panel, box, sfx=False)
            elif category == "sfx" and b.rng.random() < 0.5:
                ok = b.art_region(panel, box, sfx=True)
            else:
                ok = b.bubble_region(panel, box)
            if ok:
                break
        if category == "tails_overlap" and b.placed and b.placed[-1].panel == panel:
            for _attempt in range(10):  # a second, overlapping bubble in the same panel
                if b.bubble_region(panel, box):
                    break
    order = _panel_order(direction)
    sign = -1 if direction == "rtl" else 1
    placed = sorted(b.placed, key=lambda p: (order.index(p.panel), p.y, sign * p.x))
    regions = []
    for n, p in enumerate(placed, start=1):
        ys, xs = np.nonzero(p.text_mask)
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        region_cat = category if not (category == "sfx" and p.type != "sfx") else "flat_white"
        ref = (refs or {}).get(p.source_id, {})
        regions.append({
            "region_id": f"{page_id}-r{n:02d}", "category": region_cat, "type": p.type,
            "lang": p.lang, "text": p.text, "vertical": p.vertical,
            "text_polygon": [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
            "text_mask": rle.encode(p.text_mask),
            "bubble_mask": rle.encode(p.bubble) if p.bubble is not None else None,
            "safe_mask": rle.encode(p.safe), "reading_order": n, "font": p.font,
            "font_size_px": float(p.size), "background": p.background,
            "source_id": p.source_id,
            "references_ar": list(ref.get(p.lang, ref.get("all", []))),
            "reference_kind": "silver" if ref else "none",
        })  # fmt: skip
    gt = {
        "page_id": page_id, "series_id": f"synthetic-{split}", "split": split,
        "categories": [category], "lang": lang, "reading_direction": direction,
        "width": W, "height": H, "seed": seed, "generator": GENERATOR_VERSION,
        "regions": regions,
    }  # fmt: skip
    return b.page, b.clean, gt
