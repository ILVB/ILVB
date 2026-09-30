"""Edge cases E-02, E-03, E-04, E-06, E-08, E-10, E-11, E-15, E-16, E-18 (v0.1.0 behaviour).

Phase 0 pins what v0.1.0 already does; the owner steps in docs/EDGE_CASE_MATRIX.md
extend these tests when Phase 1 changes the behaviour.
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from manga_ar.config import load_config
from manga_ar.detect.base import TextBlock
from manga_ar.detect.bubble import FloodBubbleSegmenter
from manga_ar.detect.geometry import ellipse_mask
from manga_ar.detect.reading_order import assign_reading_order, default_mode
from manga_ar.errors import ModelUnavailableError
from manga_ar.io.loader import load_image_bytes
from manga_ar.io.tiling import needs_tiling
from manga_ar.models.device import run_with_cpu_fallback
from manga_ar.models.manager import ModelManager
from manga_ar.schemas import BBox, CropMask, Flag, Region, RegionType
from manga_ar.translate.normalize_ar import normalize_ar
from manga_ar.typeset.arabic_text import shape_line
from manga_ar.typeset.contrast import WHITE
from manga_ar.typeset.fonts import FontRegistry
from manga_ar.typeset.layout import Typesetter
from manga_ar.typeset.render import render_layer

CFG = load_config(environ={})
TS = Typesetter(CFG.typeset, FontRegistry())


def _bubble(w: int, h: int, x0: int = 20, y0: int = 20, fill: tuple[int, int, int] = (255,) * 3
            ) -> Region:  # fmt: skip
    box = BBox(x0, y0, x0 + w, y0 + h)
    return Region(id="r", type=RegionType.BUBBLE, bbox=box, fill_color=fill,
                  bubble_mask=CropMask(box, ellipse_mask((h, w), BBox(0, 0, w, h))))  # fmt: skip


def test_e02_text_on_artwork_gets_an_outline() -> None:
    art = np.full((300, 400, 3), 128, np.uint8)
    art[::6] = 40  # hatching: a busy background
    region = Region(id="f", type=RegionType.FREE_TEXT, bbox=BBox(150, 130, 250, 160),
                    lines=[BBox(150, 130, 250, 160)])  # fmt: skip
    p = TS.layout(region, "صوت في الظلام", (300, 400), art)
    assert p.outline_rgb is not None and p.outline_px >= 1


def test_e03_touching_bubbles_stay_separate() -> None:
    img = np.full((240, 400, 3), 255, np.uint8)
    for cx in (110, 280):  # outlines touch at x ≈ 195
        cv2.ellipse(img, (cx, 120), (86, 90), 0, 0, 360, (0, 0, 0), 2)
    blocks = []
    for x0 in (90, 260):
        img[105:135, x0 : x0 + 40] = 0
        box = BBox(x0, 105, x0 + 40, 135)
        blocks.append(TextBlock(box, [box], text_mask=CropMask(box, np.ones((30, 40), bool)),
                                glyph_size=30))  # fmt: skip
    regions = FloodBubbleSegmenter(CFG.detect).segment(img, blocks, "p")
    assert len(regions) == 2 and not any(Flag.MERGED in r.flags for r in regions)
    a, b = (r.bubble_mask.to_full(240, 400) for r in regions)  # type: ignore[union-attr]
    assert not (a & b).any()  # never one shared bubble
    ordered = assign_reading_order(list(regions), img, "manga_rtl")
    assert [r.bbox.x0 for r in ordered] == [260, 90]  # right bubble first, stable
    assert [r.bbox.x0 for r in assign_reading_order(list(regions), img, "manga_rtl")] == [260, 90]


def test_e04_vertical_source_is_typeset_horizontally() -> None:
    region = _bubble(120, 320)
    region.vertical = True
    p = TS.layout(region, "هذا نص طويل كان عموديا في الأصل", (360, 160))
    assert len(p.lines) >= 2 and all(ln.width > 0 for ln in p.lines)
    ys = [ln.y for ln in p.lines]
    assert ys == sorted(ys) and len(set(ys)) == len(ys)  # stacked horizontal lines


def test_e06_symbols_survive_and_never_render_as_tofu() -> None:
    text = normalize_ar("أحبك ♡ جدا ★ ♪", "western", [])
    assert all(s in text for s in "♡★♪")
    p = TS.layout(_bubble(260, 300), text, (340, 300))
    assert "".join(ln.logical for ln in p.lines).count("♡") == 1
    runs = [rf for ln in p.lines for rf in p.engine.runs(ln.visual)]  # type: ignore[union-attr]
    for run, face in runs:
        assert all(ch.isspace() or ord(ch) in face.cmap for ch in run)  # no tofu
    symbols = {ch for run, face in runs if face.role == "symbol" for ch in run}
    assert symbols == {"♡", "★", "♪"}  # the fallback fonts carry them
    assert not p.engine.missing  # type: ignore[union-attr]


def test_e08_too_long_text_is_flagged_never_clipped() -> None:
    long = "لن أسامحك أبدًا على ما فعلته بي وبأصدقائي، وسأجعلك تدفع الثمن غاليًا مهما طال الزمن!"
    region = _bubble(60, 50)
    p = TS.layout(region, long, (200, 200), np.full((200, 200, 3), 255, np.uint8))
    assert p.overflow and Flag.OVERFLOW_RISK in region.flags and p.ladder[-1] == "hard-floor"
    layer = render_layer((200, 200), [p])
    assert (layer[..., 3] > 0).any() and p.ink is not None
    assert p.ink.x0 >= 0 and p.ink.y0 >= 0 and p.ink.x1 <= 200 and p.ink.y1 <= 200


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def test_e10_image_modes_become_rgb_uint8_and_tall_strips_tile() -> None:
    g16 = Image.fromarray(np.linspace(0, 65535, 64 * 64).reshape(64, 64).astype(np.uint16))
    pal = Image.new("P", (64, 64), 3)
    pal.putpalette([10, 20, 30] * 256)
    rgba = Image.new("RGBA", (64, 64), (200, 10, 10, 0))
    buf = io.BytesIO()
    Image.new("CMYK", (64, 64), (0, 255, 255, 0)).save(buf, "JPEG", quality=95)
    for data in (_png(g16), _png(pal), _png(rgba), buf.getvalue()):
        rgb = load_image_bytes(data, "x.png").rgb
        assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
    assert needs_tiling(20000, 800, CFG.tiling.aspect_trigger, CFG.tiling.height_trigger)
    assert not needs_tiling(1400, 1000, CFG.tiling.aspect_trigger, CFG.tiling.height_trigger)


def test_e11_oom_falls_back_to_cpu_and_offline_missing_model_is_actionable(
    tmp_path: Path,
) -> None:
    def run(device: str) -> str:
        if device == "cuda":
            raise RuntimeError("CUDA out of memory. Tried to allocate 1 GiB")
        return "done"

    assert run_with_cpu_fallback(run, "cuda", "lama") == ("done", "cpu")
    mgr = ModelManager(tmp_path, offline=True)
    with pytest.raises(ModelUnavailableError, match="offline") as exc:
        mgr.ensure("lama")
    message = str(exc.value)  # actionable: where to get it, where to put it, its checksum
    assert "https://" in message and "sha256" in message and str(tmp_path) in message


def test_e15_digit_policy_and_mixed_direction() -> None:
    assert "١٠" in normalize_ar("انتظر 10 ثوان", "arabic_indic", [])
    assert "10" in normalize_ar("انتظر 10 ثوان", "western", [])
    visual = shape_line("انتظر 10 ثوان")
    assert "10" in visual and "01" not in visual  # digits keep LTR order inside RTL text


def test_e16_reading_direction_rtl_vs_ltr() -> None:
    rgb = np.full((400, 600, 3), 255, np.uint8)
    regions = [Region(id=f"r{i}", type=RegionType.BUBBLE, bbox=BBox(x, 100, x + 80, 140))
               for i, x in enumerate((50, 450))]  # fmt: skip
    rtl = assign_reading_order(list(regions), rgb, "manga_rtl")
    ltr = assign_reading_order(list(regions), rgb, "comic_ltr")
    assert [r.bbox.x0 for r in rtl] == [450, 50] and [r.bbox.x0 for r in ltr] == [50, 450]
    assert (default_mode("ja"), default_mode("zh"), default_mode("ko")) == (
        "manga_rtl", "comic_ltr", "webtoon_ttb")  # fmt: skip


def test_e18_dark_bubble_gets_light_text() -> None:
    p = TS.layout(_bubble(240, 300, fill=(15, 15, 15)), "لا تقترب", (340, 280))
    assert p.text_rgb == WHITE
