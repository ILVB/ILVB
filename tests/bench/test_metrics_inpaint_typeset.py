"""Inpainting, typesetting and statistics metrics against hand-computed values."""

from __future__ import annotations

import math

import numpy as np
import pytest

from benchmarks.metrics import inpaint, stats, typeset


def _img(h: int = 20, w: int = 20, v: int = 200) -> np.ndarray:
    return np.full((h, w, 3), v, np.uint8)


def test_changed_outside_respects_two_px_ring() -> None:
    orig = _img()
    declared = np.zeros((20, 20), bool)
    declared[10, 10] = True
    erased = orig.copy()
    erased[10, 12] = 0  # 2 px from the mask: inside the feather ring
    erased[10, 13] = 0  # 3 px: outside
    erased[0, 0, 1] = 0  # any channel counts
    assert inpaint.changed_outside(orig, erased, declared) == 2
    assert inpaint.changed_outside(orig, orig, declared) == 0


def test_crop_box_clips_margin() -> None:
    m = np.zeros((20, 30), bool)
    m[2:5, 25:28] = True
    assert inpaint.crop_box(m, 8) == (17, 0, 30, 13)


def test_ssim_psnr_hand_values() -> None:
    clean = np.zeros((40, 40, 3), np.uint8)
    clean[:, 20:] = 100
    assert inpaint.ssim_psnr(clean, clean, (0, 0, 40, 40)) == (1.0, inpaint.PSNR_CAP)
    erased = clean.copy()
    erased[:20] += 10  # MSE over the whole crop = 100 * 0.5 = 50
    ssim, psnr = inpaint.ssim_psnr(erased, clean, (0, 0, 40, 40))
    assert psnr == pytest.approx(10 * math.log10(255**2 / 50))
    assert 0.0 < ssim < 1.0


def test_pixel_residual() -> None:
    clean = _img(40, 40, 250)
    mask = np.zeros((40, 40), bool)
    mask[10:30, 10:30] = True
    assert not inpaint.pixel_residual(clean, clean, mask)
    left = clean.copy()
    left[15:25, 18:21] = 20  # a dark stroke left behind (30 px)
    assert inpaint.pixel_residual(left, clean, mask)
    faint = clean.copy()
    faint[15:25, 18:21] = 230  # below tolerance: noise, not lettering
    assert not inpaint.pixel_residual(faint, clean, mask)


def test_ink_outside_and_available_width() -> None:
    safe = np.zeros((10, 20), bool)
    safe[2:8, 2:18] = True
    safe[4, 2:18] = False
    safe[4, 2:5] = safe[4, 8:13] = True  # runs of 3 and 5 on row 4
    ink = np.zeros_like(safe)
    ink[3, 1:4] = True  # (3,1) outside
    assert typeset.ink_outside(ink, safe) == 1
    assert typeset.available_width(safe, 2, 4) == 16.0
    assert typeset.available_width(safe, 2, 5) == 5.0  # min over rows 2..4
    assert typeset.available_width(safe, 0, 3) == 0.0  # row 0 empty
    assert typeset.available_width(safe, 5, 5) == 0.0


def test_raggedness_orphan_hyphen_floor() -> None:
    safe = np.zeros((30, 12), bool)
    safe[:, 1:11] = True  # width 10 everywhere
    boxes = [[1, 0, 9, 10], [1, 10, 11, 20], [1, 20, 4, 30]]  # slacks 0.2, 0.0; last ignored
    assert typeset.raggedness(boxes, safe) == pytest.approx((0.2**2 + 0.0) / 2)
    assert typeset.raggedness(boxes[:1], safe) is None
    assert typeset.is_orphan(["مرحبا بك في", "البيت"])
    assert not typeset.is_orphan(["مرحبا بك", "في البيت"])
    assert not typeset.is_orphan(["وحيد"])
    assert typeset.arabic_hyphen_breaks(["مرح-", "با", "كلمة-"]) == 1  # last line ignored
    assert typeset.above_floor(15.4, 1400)
    assert not typeset.above_floor(15.3, 1400)


def test_bootstrap_ci_and_percentiles() -> None:
    lo, hi = stats.bootstrap_ci([3.0] * 10, lambda xs: float(np.mean(xs)))
    assert (lo, hi) == (3.0, 3.0)
    items = list(range(100))
    a = stats.bootstrap_ci(items, lambda xs: float(np.mean(xs)), n=300)
    assert a == stats.bootstrap_ci(items, lambda xs: float(np.mean(xs)), n=300)  # seeded
    assert a[0] < 49.5 < a[1]
    assert all(math.isnan(v) for v in stats.bootstrap_ci([], len))
    p = stats.percentiles([float(v) for v in range(1, 101)])
    assert p == {"p50": 50.5, "p95": pytest.approx(95.05), "max": 100.0}


@pytest.mark.integration
def test_probe_residual_detects_text() -> None:
    from PIL import Image, ImageDraw, ImageFont

    from benchmarks.generators.fonts import font_path

    canvas = Image.new("RGB", (320, 120), (255, 255, 255))
    font = ImageFont.truetype(str(font_path("comic_neue_bold")), 48)
    ImageDraw.Draw(canvas).text((10, 30), "HELLO", font=font, fill=(0, 0, 0))
    img = np.asarray(canvas, dtype=np.uint8)
    assert inpaint.probe_residual(img, (0, 0, 320, 120))
    assert not inpaint.probe_residual(_img(120, 320, 255), (0, 0, 320, 120))
