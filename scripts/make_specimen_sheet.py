"""ARVS L8: visual specimen sheets for every bundled Arabic font (manual inspection).

Usage: python scripts/make_specimen_sheet.py [--out .cache/specimen] [--fonts KEY ...]

One PNG per font: the canonical corpus (tests/data/arabic_corpus.json) right-aligned at
40 px with its check label, then four bubble shapes (round, tall, wide, rectangular
narration) and a free-text box laid out by the Typesetter with a long line, a question
and a bracket/digit mix. Bubble outlines and the layout masks are drawn so fit and
padding can be judged. ``overview.png`` stacks thumbnails of all sheets.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from manga_ar.config import load_config
from manga_ar.detect.geometry import ellipse_mask
from manga_ar.schemas import BBox, CropMask, Region, RegionType
from manga_ar.translate.normalize_ar import normalize_ar
from manga_ar.typeset.arabic_text import strip_harakat
from manga_ar.typeset.fonts import FontRegistry
from manga_ar.typeset.layout import Typesetter
from manga_ar.typeset.render import composite, render_layer

ROOT = Path(__file__).resolve().parents[1]
WIDTH = 1900
MARGIN = 30
LINE_SIZE = 40
BUBBLE_TEXTS = [
    "لن أسامحك أبدًا على ما فعلته بي وبأصدقائي، وسأجعلك تدفع الثمن غاليًا مهما طال الزمن!",
    "هل أنت بخير؟ لقد كنت قلقًا عليك طوال الليل",
    "اضغط على (OK) ثم انتظر 10 ثوانٍ",
    "في تلك الليلة، تغير كل شيء إلى الأبد…",
    "كلا، لا يمكنني ذلك ♡",
]


def _label_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size)


def _bubble_region(kind: str, x0: int, y0: int, w: int, h: int) -> Region:
    box = BBox(x0, y0, x0 + w, y0 + h)
    if kind == "narration":
        mask = np.ones((h, w), bool)
        return Region(
            id=kind,
            type=RegionType.NARRATION,
            bbox=box,
            fill_color=(255, 255, 255),
            bubble_mask=CropMask(box, mask),
        )
    return Region(
        id=kind,
        type=RegionType.BUBBLE,
        bbox=box,
        fill_color=(255, 255, 255),
        bubble_mask=CropMask(box, ellipse_mask((h, w), BBox(0, 0, w, h))),
    )


def _corpus() -> list[tuple[str, str]]:
    data = json.loads((ROOT / "tests" / "data" / "arabic_corpus.json").read_text("utf-8"))
    rows = [(item["text"], item["checks"]) for item in data["canonical"]]
    rows += [(text, "extra") for text in data["extra"]]
    return rows


def font_sheet(ts: Typesetter, key: str) -> tuple[Image.Image, list[str]]:
    """Specimen for one font; returns the image and notes (ladder use, overflow)."""
    corpus = _corpus()
    line_h = int(LINE_SIZE * 1.6)
    corpus_h = MARGIN * 2 + 60 + line_h * len(corpus)
    shapes = [
        ("round", 300, 240),
        ("tall", 200, 330),
        ("wide", 420, 170),
        ("narration", 330, 150),
        ("free", 300, 120),
    ]
    bubbles_h = 360 + MARGIN * 2
    height = corpus_h + bubbles_h * 2
    img = Image.new("RGB", (WIDTH, height), "white")
    draw = ImageDraw.Draw(img)
    face = ts.registry.get(key)
    head = f"{key} - {face.family} ({face.license}), path: {ts.cfg.render_path}"
    draw.text((MARGIN, MARGIN), head, fill=(0, 0, 160), font=_label_font(26))
    notes: list[str] = []
    y = MARGIN + 60
    small = _label_font(16)
    for text, checks in corpus:
        logical = normalize_ar(text, ts.cfg.digits)
        if ts.cfg.strip_harakat:
            logical = strip_harakat(logical)
        engine = ts.engine_for(logical, key)
        visual = engine.shape(logical, ts.cfg.strip_harakat)
        width = engine.measure(visual, LINE_SIZE)
        if width > WIDTH - 2 * MARGIN - 260:  # long sentence: shrink to fit the sheet
            size = int(LINE_SIZE * (WIDTH - 2 * MARGIN - 260) / width)
        else:
            size = LINE_SIZE
        width = engine.measure(visual, size)
        engine.draw(draw, (WIDTH - MARGIN - width, y), visual, size, (0, 0, 0, 255))
        draw.text((MARGIN, y + 10), checks[:34], fill=(120, 120, 120), font=small)
        draw.line(
            [(MARGIN, y + line_h - 6), (WIDTH - MARGIN, y + line_h - 6)], fill=(235, 235, 235)
        )
        y += line_h
    # Bubble rows: each text in each shape (two rows: long texts, short texts).
    page = np.asarray(img).copy()
    placements = []
    for row, texts in enumerate((BUBBLE_TEXTS[:3], BUBBLE_TEXTS[2:])):
        x = MARGIN
        top = corpus_h + row * bubbles_h + MARGIN
        for i, (kind, w, h) in enumerate(shapes):
            text = texts[i % len(texts)]
            if kind == "free":
                region = Region(
                    id="free",
                    type=RegionType.FREE_TEXT,
                    bbox=BBox(x, top + 80, x + w, top + 80 + h),
                    lines=[BBox(x, top + 80, x + w, top + 80 + h)],
                )
                page[top + 60 : top + 220, x - 10 : x + w + 10] = (210, 180, 150)  # artwork
                cv2.line(page, (x, top + 70), (x + w, top + 210), (120, 80, 60), 3)
            else:
                region = _bubble_region(kind, x, top, w, h)
                full = region.bubble_mask.to_full(page.shape[0], page.shape[1])  # type: ignore[union-attr]
                outline = cv2.dilate(full.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                page[outline & ~full] = (0, 0, 0)
            p = ts.layout(region, text, page.shape[:2], page)
            placements.append(p)
            geom = p.geometry
            if geom is not None:  # layout mask edge, light blue
                m = np.zeros(page.shape[:2], np.uint8)
                b = geom.box.clip(page.shape[1], page.shape[0])
                m[b.y0 : b.y1, b.x0 : b.x1] = geom.mask[: b.height, : b.width]
                edge = (m > 0) & ~(cv2.erode(m, np.ones((3, 3), np.uint8)) > 0)
                page[edge] = (150, 200, 255)
            label = f"{kind}: {p.size}px, {len(p.lines)} lines, {p.strategy}"
            if p.ladder:
                label += " [" + ",".join(p.ladder) + "]"
                notes.append(f"{kind}/{text[:12]}: ladder {p.ladder}")
            if p.overflow:
                notes.append(f"{kind}/{text[:12]}: OVERFLOW")
            cv2.putText(
                page,
                label[:48],
                (x, top + max(h, 230) + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (90, 90, 90),
                1,
                cv2.LINE_AA,
            )
            x += w + 50
    out = composite(page, render_layer(page.shape[:2], placements))
    return Image.fromarray(out), notes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / ".cache" / "specimen")
    ap.add_argument("--fonts", nargs="*", default=None)
    ap.add_argument("--render-path", choices=["basic", "raqm"], default="basic")
    args = ap.parse_args(argv)
    cfg = load_config(environ={})
    registry = FontRegistry(include_system=False)
    keys = args.fonts or [k for k, f in registry.fonts.items() if f.role == "arabic"]
    args.out.mkdir(parents=True, exist_ok=True)
    thumbs = []
    for key in keys:
        face = registry.get(key)
        path = args.render_path
        if path == "basic" and not face.basic_ok:
            path = "raqm"  # fonts without presentation forms can only use HarfBuzz shaping
        ts = Typesetter(dataclasses.replace(cfg.typeset, font=key, render_path=path), registry)
        if path == "raqm" and not ts.raqm:
            sys.stdout.write(f"{key}: skipped (needs libraqm; BASIC coverage incomplete)\n")
            continue
        sheet, notes = font_sheet(ts, key)
        target = args.out / f"specimen_{key}.png"
        sheet.save(target)
        thumbs.append(sheet.resize((sheet.width // 3, sheet.height // 3)))
        sys.stdout.write(f"{target}  {'; '.join(notes) if notes else 'no ladder steps'}\n")
    if thumbs:
        cols = 4
        tw, th = max(t.width for t in thumbs), max(t.height for t in thumbs)
        rows = (len(thumbs) + cols - 1) // cols
        overview = Image.new("RGB", (cols * tw, rows * th), "white")
        for i, t in enumerate(thumbs):
            overview.paste(t, ((i % cols) * tw, (i // cols) * th))
        overview.save(args.out / "overview.png")
        sys.stdout.write(f"{args.out / 'overview.png'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
