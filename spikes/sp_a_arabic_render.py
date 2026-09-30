"""SP-A: Arabic rendering feasibility (reshape + bidi + BASIC vs. RAQM).

Renders a set of strings in two fonts through both render paths, writes one PNG and
prints connected-component counts for the joining oracles (محمد ⇒ 1, الحمد ⇒ 2).
Run: .venv/bin/python spikes/sp_a_arabic_render.py
"""

from __future__ import annotations

from pathlib import Path

import arabic_reshaper
import cv2
import numpy as np
from bidi import get_display
from PIL import Image, ImageDraw, ImageFont, features

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "spikes" / "out"
FONTS = [
    ROOT / "src/manga_ar/assets/fonts/NotoNaskhArabic-Variable.ttf",
    ROOT / "src/manga_ar/assets/fonts/Tajawal-Regular.ttf",
]
SYMBOL_FONT = ROOT / "src/manga_ar/assets/fonts/NotoSansSymbols2-Regular.ttf"
STRINGS = [
    "مرحبا بالعالم",
    "لا تذهب!",
    "هل أنت بخير؟",
    "الحمد لله",
    "محمد",
    "سأنتظرك عند الساعة 5 مساء",
    "اضغط على (OK) ثم انتظر 10 ثوان",
    "كلا، لا يمكنني ذلك ♡",
]

RESHAPER = arabic_reshaper.ArabicReshaper(
    configuration={
        "delete_harakat": True,
        "support_ligatures": True,
        "ARABIC LIGATURE ALLAH": False,
    }
)


def visual(text: str) -> str:
    return str(get_display(RESHAPER.reshape(text), base_dir="R"))


def components(img: Image.Image) -> int:
    arr = np.asarray(img.convert("L"))
    binary = (arr < 128).astype(np.uint8)
    n, _ = cv2.connectedComponents(binary, connectivity=8)
    return int(n - 1)


def render_line(text: str, font_path: Path, size: int, raqm: bool) -> Image.Image:
    layout = ImageFont.Layout.RAQM if raqm else ImageFont.Layout.BASIC
    font = ImageFont.truetype(str(font_path), size, layout_engine=layout)
    kwargs = {"direction": "rtl", "language": "ar"} if raqm else {}
    shown = text if raqm else visual(text)
    left, top, right, bottom = font.getbbox(shown, **kwargs)
    img = Image.new("L", (right - left + 40, bottom - top + 40), 255)
    ImageDraw.Draw(img).text((20 - left, 20 - top), shown, font=font, fill=0, **kwargs)
    return img


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raqm_ok = features.check("raqm")
    print(f"RAQM available: {raqm_ok}")
    rows: list[Image.Image] = []
    for font_path in FONTS:
        for s in STRINGS:
            p1 = render_line(s, font_path, 64, raqm=False)
            p2 = render_line(s, font_path, 64, raqm=True) if raqm_ok else p1
            row = Image.new("L", (p1.width + p2.width + 60, max(p1.height, p2.height)), 255)
            row.paste(p1, (0, 0))
            row.paste(p2, (p1.width + 60, 0))
            rows.append(row)
    sheet = Image.new("L", (max(r.width for r in rows), sum(r.height for r in rows)), 255)
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height
    sheet.save(OUT / "sp_a_arabic.png")
    print(f"wrote {OUT / 'sp_a_arabic.png'} {sheet.size}")

    for font_path in FONTS:
        for word, expected in (("محمد", 1), ("الحمد", 2)):
            c_basic = components(render_line(word, font_path, 96, raqm=False))
            c_raqm = components(render_line(word, font_path, 96, raqm=True)) if raqm_ok else -1
            # Negative control: raw logical text through BASIC (unshaped, not reordered).
            font = ImageFont.truetype(str(font_path), 96, layout_engine=ImageFont.Layout.BASIC)
            l, t, r, b = font.getbbox(word)
            neg = Image.new("L", (r - l + 40, b - t + 40), 255)
            ImageDraw.Draw(neg).text((20 - l, 20 - t), word, font=font, fill=0)
            c_neg = components(neg)
            print(
                f"{font_path.name:32s} {word}: basic={c_basic} raqm={c_raqm} "
                f"unshaped_negative={c_neg} expected={expected}"
            )
    for s in STRINGS[:3]:
        print(repr(s), "->", " ".join(f"U+{ord(c):04X}" for c in visual(s)))


if __name__ == "__main__":
    main()
