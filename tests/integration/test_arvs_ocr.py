"""ARVS L7: OCR round-trip of rendered Arabic (EasyOCR Arabic model from GitHub).

Each bundled BASIC-path font renders ≥ 20 phrases plainly and inside a laid-out bubble;
the text must read back with normalised Levenshtein similarity ≥ 0.8 for ≥ 90 % of
samples overall (≥ 75 % per font; calibrated in DECISIONS D-031). Reversed/unshaped
renders must score markedly lower.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from manga_ar.config import load_config
from manga_ar.detect.geometry import ellipse_mask
from manga_ar.metrics import similarity
from manga_ar.schemas import BBox, CropMask, Region, RegionType
from manga_ar.translate.normalize_ar import normalize_ar
from manga_ar.typeset.arabic_text import reshaper_for
from manga_ar.typeset.fonts import FontRegistry, load_font
from manga_ar.typeset.layout import Typesetter
from manga_ar.typeset.render import composite, render_layer
from manga_ar.typeset.textline import FontChoice

pytestmark = pytest.mark.integration
PHRASES = [
    "مرحبا بالعالم",
    "لا تذهب",
    "هل أنت بخير",
    "الحمد لله",
    "بلا شك ولا تردد",
    "انتظر قليلا",
    "كلا لا يمكنني ذلك",
    "ماذا حدث",
    "سأعود قريبا",
    "أين أنت الآن",
    "شكرا جزيلا لك",
    "لن أستسلم أبدا",
    "هيا بنا نذهب",
    "هذا مستحيل",
    "أنا جائع جدا",
    "من أنت",
    "اهدأ قليلا",
    "لقد تأخرنا",
    "صباح الخير",
    "إلى اللقاء",
    "أحبك كثيرا",
]
REG = FontRegistry()
FONTS = [
    f for f in REG.fonts.values() if f.role == "arabic" and f.basic_ok and f.source == "vendored"
]


@pytest.fixture(scope="module")
def reader(cache_dir: Path):  # type: ignore[no-untyped-def]
    if importlib.util.find_spec("easyocr") is None:
        pytest.skip("easyocr not installed")
    import easyocr

    try:
        return easyocr.Reader(
            ["ar", "en"],
            gpu=False,
            verbose=False,
            model_storage_directory=str(cache_dir / "models" / "easyocr"),
        )
    except (OSError, RuntimeError) as exc:
        pytest.skip(f"EasyOCR Arabic model unavailable: {exc}")


def _norm(text: str) -> str:
    return "".join(normalize_ar(text).split())


def _plain(draw_fn, text: str, size: int) -> np.ndarray:  # type: ignore[no-untyped-def]
    img = Image.new("RGB", (size * len(text) + 120, int(size * 2.2)), "white")
    draw_fn(ImageDraw.Draw(img), (40, size // 3), text, size)
    return np.asarray(img)


def _bubble(text: str, font: str) -> np.ndarray:
    page = np.full((360, 420, 3), 255, np.uint8)
    box = BBox(30, 30, 390, 330)
    region = Region(
        id="b",
        type=RegionType.BUBBLE,
        bbox=box,
        fill_color=(255, 255, 255),
        bubble_mask=CropMask(box, ellipse_mask((300, 360), BBox(0, 0, 360, 300))),
    )
    cfg = load_config(overrides={"typeset.font": font}, environ={})
    p = Typesetter(cfg.typeset, REG).layout(region, text, page.shape[:2])
    return composite(page, render_layer(page.shape[:2], [p]))


def _read(reader, image: np.ndarray) -> str:  # type: ignore[no-untyped-def]
    return " ".join(reader.readtext(image, detail=0, paragraph=True))


def test_l7_round_trip(reader) -> None:  # type: ignore[no-untyped-def]
    total = passed = 0
    report = []
    for face in FONTS:
        choice = FontChoice(face, [])

        def good(d, xy, t, s, choice=choice):  # type: ignore[no-untyped-def]
            choice.draw(d, xy, choice.shape(t), s, (0, 0, 0, 255))

        sims = []
        for k, phrase in enumerate(PHRASES):
            size = 48 + (k % 3) * 12  # 48, 60, 72 px
            sims.append(similarity(_norm(phrase), _norm(_read(reader, _plain(good, phrase, size)))))
        for phrase in PHRASES[:4]:
            sims.append(similarity(_norm(phrase), _norm(_read(reader, _bubble(phrase, face.key)))))
        ok = sum(s >= 0.8 for s in sims)
        report.append(f"{face.key}: {ok}/{len(sims)} ≥ 0.8 (mean {np.mean(sims):.2f})")
        assert ok / len(sims) >= 0.75, report[-1]
        total += len(sims)
        passed += ok
    print("\n".join(report))
    assert passed / total >= 0.90, (passed, total)


def test_l7_negative_controls_score_lower(reader) -> None:  # type: ignore[no-untyped-def]
    face = REG.get("NotoNaskhArabic")
    choice = FontChoice(face, [])
    font = lambda s: load_font(str(face.path), s)  # noqa: E731
    unshaped = lambda d, xy, t, s: d.text(xy, t, font=font(s), fill=(0, 0, 0))  # noqa: E731
    reversed_ = lambda d, xy, t, s: d.text(  # noqa: E731
        xy, reshaper_for(choice.unshaped_isolated).reshape(t), font=font(s), fill=(0, 0, 0)
    )
    for fn in (unshaped, reversed_):
        sims = [similarity(_norm(p), _norm(_read(reader, _plain(fn, p, 60)))) for p in PHRASES[:10]]
        assert float(np.mean(sims)) < 0.4, sims
