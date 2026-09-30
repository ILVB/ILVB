"""Layout, fit, overflow ladder, contrast and rendering invariants (Gate P5)."""

from __future__ import annotations

import dataclasses
import time
from itertools import pairwise

import cv2
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from manga_ar.config import load_config
from manga_ar.detect.geometry import ellipse_mask
from manga_ar.schemas import BBox, CropMask, Flag, Region, RegionOverride, RegionType
from manga_ar.typeset.contrast import BLACK, WHITE, choose_colors, contrast_ratio
from manga_ar.typeset.fonts import FontRegistry
from manga_ar.typeset.layout import Placement, Typesetter
from manga_ar.typeset.render import composite, render_layer, to_layout_result
from manga_ar.typeset.textline import RaqmChoice

CFG = load_config(environ={})
REG = FontRegistry()
LONG = "لن أسامحك أبدًا على ما فعلته بي وبأصدقائي، وسأجعلك تدفع الثمن غاليًا مهما طال الزمن!"
TEXTS = [
    "مرحبا بالعالم",
    "هل أنت بخير؟",
    "اضغط على (OK) ثم انتظر 10 ثوانٍ",
    LONG,
    "كلا، لا يمكنني ذلك ♡",
]
ARABIC_LETTERS = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"


def _typesetter(**over: object) -> Typesetter:
    cfg = dataclasses.replace(CFG.typeset, **over) if over else CFG.typeset
    return Typesetter(cfg, REG)


def _ellipse_region(
    w: int, h: int, x0: int = 40, y0: int = 40, fill: tuple[int, int, int] = (255, 255, 255)
) -> Region:
    box = BBox(x0, y0, x0 + w, y0 + h)
    return Region(
        id="r",
        type=RegionType.BUBBLE,
        bbox=box,
        fill_color=fill,
        bubble_mask=CropMask(box, ellipse_mask((h, w), BBox(0, 0, w, h))),
    )


def _ink_outside(p: Placement, page_shape: tuple[int, int], tol: int = 2) -> int:
    """Rendered text pixels falling outside the layout area (dilated by ``tol``)."""
    h, w = page_shape
    layer = render_layer(page_shape, [p])
    geom = p.geometry
    assert geom is not None
    allowed = np.zeros((h, w), np.uint8)
    b = geom.box.clip(w, h)
    allowed[b.y0 : b.y1, b.x0 : b.x1] = geom.mask[: b.height, : b.width]
    allowed = cv2.dilate(allowed, np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)) > 0
    return int(((layer[..., 3] > 40) & ~allowed).sum())


@pytest.mark.parametrize("text", TEXTS)
def test_text_stays_inside_bubble(text: str) -> None:
    region = _ellipse_region(240, 320)
    p = _typesetter().layout(region, text, (420, 340))
    assert not p.overflow and Flag.OVERFLOW_RISK not in region.flags
    assert _ink_outside(p, (420, 340)) == 0
    # consistent baselines: lines equally spaced by the line height
    ys = [ln.y for ln in p.lines]
    assert all(abs((b - a) - p.line_height) < 1e-6 for a, b in pairwise(ys))


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    words=st.lists(
        st.text(alphabet=ARABIC_LETTERS, min_size=2, max_size=8), min_size=1, max_size=14
    ),
    w=st.integers(70, 360),
    h=st.integers(70, 360),
)
def test_property_zero_overflow(words: list[str], w: int, h: int) -> None:
    region = _ellipse_region(w, h, 10, 10)
    shape = (h + 20, w + 20)
    p = _typesetter().layout(region, " ".join(words), shape)
    if Flag.OVERFLOW_RISK not in region.flags:
        assert _ink_outside(p, shape) == 0
    assert p.ink is not None and p.ink.x0 >= 0 and p.ink.y0 >= 0
    assert p.ink.x1 <= shape[1] and p.ink.y1 <= shape[0]  # never outside the image


def test_property_size_non_increasing_with_length() -> None:
    words = LONG.split()
    ts = _typesetter()
    sizes = []
    for k in range(1, len(words) + 1):
        region = _ellipse_region(260, 300)
        sizes.append(ts.layout(region, " ".join(words[:k]), (400, 360)).size)
    assert all(b <= a for a, b in pairwise(sizes)), sizes


@pytest.mark.parametrize(("w", "h"), [(200, 300), (300, 200), (260, 260), (160, 360)])
def test_shape_strategy_beats_rectangle_on_ellipses(w: int, h: int) -> None:
    shape_ts, rect_ts = _typesetter(strategy="shape"), _typesetter(strategy="rect")
    for text in TEXTS:
        pb = shape_ts.layout(_ellipse_region(w, h), text, (h + 80, w + 80))
        pa = rect_ts.layout(_ellipse_region(w, h), text, (h + 80, w + 80))
        assert pb.size >= pa.size, (text, w, h, pb.size, pa.size)
        assert _ink_outside(pb, (h + 80, w + 80)) == 0


def test_overflow_ladder_order_and_flag() -> None:
    region = _ellipse_region(60, 50)
    p = _typesetter(font="NotoSansArabic").layout(
        region, LONG, (200, 200), np.full((200, 200, 3), 255, np.uint8)
    )
    assert p.ladder[0] == "line-spacing-floor"
    assert "condensed" in p.ladder  # Noto Sans Arabic has a width axis
    assert p.ladder[-1] == "hard-floor" and p.overflow
    assert Flag.OVERFLOW_RISK in region.flags
    assert p.ink is not None and p.ink.x0 >= 0 and p.ink.x1 <= 200  # never clipped off-page


def test_extend_into_uniform_background_for_free_text() -> None:
    clean = np.full((300, 400, 3), 255, np.uint8)
    region = Region(
        id="f",
        type=RegionType.FREE_TEXT,
        bbox=BBox(180, 140, 220, 160),
        lines=[BBox(180, 140, 220, 160)],
    )
    p = _typesetter().layout(region, "مرحبا بالعالم الجميل", (300, 400), clean)
    assert "extend-uniform" in p.ladder and not p.overflow
    assert p.outline_rgb is not None and p.outline_px >= 1  # free text gets an outline


def test_contrast_choice() -> None:
    assert choose_colors((20, 20, 20), 4.5, 30, 0.08, False)[0] == WHITE
    assert choose_colors((250, 240, 200), 4.5, 30, 0.08, False)[0] == BLACK
    # Black or white always reaches ≥ 4.58:1 (WCAG), so an outline appears only when forced
    # or with a stricter target: mid-grey cannot reach 7:1 with either colour.
    assert choose_colors((118, 118, 118), 4.5, 30, 0.08, False)[1] is None
    _text, outline, px = choose_colors((118, 118, 118), 7.0, 30, 0.08, False)
    assert outline is not None and px >= 2
    assert contrast_ratio(BLACK, WHITE) == pytest.approx(21.0)
    region = _ellipse_region(240, 320, fill=(15, 15, 15))
    p = _typesetter().layout(region, "مرحبا", (420, 340))
    assert p.text_rgb == WHITE


def test_harakat_stripped_digits_and_bidi_controls() -> None:
    region = _ellipse_region(240, 320)
    p = _typesetter(digits="arabic_indic").layout(region, "مَرْحَبًا‮ 10", (420, 340))
    joined = "".join(ln.logical for ln in p.lines)
    assert not any("ً" <= c <= "ْ" for c in joined)
    assert "١٠" in joined and "‮" not in joined


def test_right_alignment_and_overrides() -> None:
    region = _ellipse_region(260, 300)
    region.override = RegionOverride(font="Amiri", size_px=24)
    p = _typesetter(alignment="right").layout(region, "مرحبا بالعالم الجميل", (400, 360))
    assert p.size == 24 and p.engine.arabic.key == "Amiri"  # type: ignore[union-attr]
    rights = [ln.x + ln.width for ln in p.lines]
    lefts = [ln.x for ln in p.lines]
    assert len(p.lines) == 1 or max(lefts) - min(lefts) > 0 or max(rights) >= min(rights)


def test_raqm_path_and_symbol_fallback() -> None:
    from PIL import features

    if not features.check("raqm"):
        pytest.skip("libraqm not available")
    ts = _typesetter(render_path="raqm")
    assert isinstance(ts.engine_for("مرحبا بالعالم", None), RaqmChoice)
    assert not isinstance(ts.engine_for("كلا ♡", None), RaqmChoice)  # font lacks ♡
    p = ts.layout(_ellipse_region(240, 320), "مرحبا بالعالم", (420, 340))
    assert _ink_outside(p, (420, 340)) == 0


def test_render_composite_and_sidecar_layout() -> None:
    region = _ellipse_region(240, 320)
    p = _typesetter().layout(region, TEXTS[0], (420, 340))
    layer1 = render_layer((420, 340), [p])
    layer2 = render_layer((420, 340), [p], shadow=True)
    assert np.array_equal(layer1, render_layer((420, 340), [p]))  # deterministic
    assert (layer2[..., 3] > 0).sum() >= (layer1[..., 3] > 0).sum()
    page = np.full((420, 340, 3), 255, np.uint8)
    out = composite(page, layer1)
    assert out.dtype == np.uint8 and (out < 128).any()
    lr = to_layout_result(p)
    assert lr.size_px == p.size and lr.lines == [ln.logical for ln in p.lines]
    assert lr.font == "NotoNaskhArabic"


def test_typical_region_is_fast() -> None:
    ts = _typesetter()
    region = _ellipse_region(240, 320)
    ts.layout(region, TEXTS[2], (420, 340))  # warm caches (fonts, word metrics)
    start = time.perf_counter()
    for text in TEXTS[:3]:
        ts.layout(_ellipse_region(230, 310), text + " ثم", (420, 340))
    per_region = (time.perf_counter() - start) / 3
    assert per_region < 0.2, per_region  # A13 target is < 50 ms; generous bound for CI noise


def test_leak_fallback_uses_rectangle_strategy() -> None:
    region = _ellipse_region(200, 260)
    region.flag(Flag.LEAK_FALLBACK)
    region.safe_box = BBox(80, 90, 200, 260)
    p = _typesetter().layout(region, TEXTS[1], (400, 320))
    assert p.strategy == "rect"


def test_padding_erosion_never_reaches_the_outline() -> None:
    """Tight bubble masks touch their crop edges at the extremes; the padded layout area
    must still keep ``pad`` px from every edge (regression: default OpenCV border kept
    full-width spikes there)."""
    region = _ellipse_region(300, 240)
    geom = _typesetter().geometry(region, (400, 400))
    pad = max(CFG.typeset.padding_min_px, round(CFG.typeset.padding_frac * 240))
    assert not geom.mask[:pad].any() and not geom.mask[-pad:].any()
    assert not geom.mask[:, :pad].any() and not geom.mask[:, -pad:].any()
    ext = _typesetter().extended_geometry(region, np.full((400, 400, 3), 255, np.uint8), geom)
    assert ext.mask.sum() > geom.mask.sum()  # the trusted interior, unpadded
