"""Inpainting invariants (Gate P3). Model-free: LaMa is replaced by a deterministic fake."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import cv2
import numpy as np
import pytest

from manga_ar import synth
from manga_ar.config import load_config
from manga_ar.detect.bubble import FloodBubbleSegmenter
from manga_ar.detect.classical import ClassicalDetector
from manga_ar.errors import InpaintError
from manga_ar.inpaint.mask import build_mask, halo_mask, stray_ink, stroke_width
from manga_ar.inpaint.residual import ResidualChecker
from manga_ar.inpaint.strategy import Background, RegionInpainter, classify_background
from manga_ar.schemas import BBox, CropMask, Flag, Region, RegionType

CFG = load_config(environ={})


class FakeLama:
    """Deterministic stand-in: fills the mask with the median of the crop."""

    name = "lama"

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def available(self) -> bool:
        return True

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        self.calls += 1
        if self.fail:
            raise InpaintError("simulated CUDA OOM")
        out = image.copy()
        out[mask] = np.median(image[~mask], axis=0).astype(np.uint8)
        return out


def _inpainter(strategy: str = "auto", lama: FakeLama | None = None) -> RegionInpainter:
    cfg = dataclasses.replace(CFG.inpaint, strategy=strategy)
    return RegionInpainter(cfg, lama if lama is not None else FakeLama())


@pytest.fixture(scope="module")
def pages(cjk_ready: Path, cache_dir: Path) -> list[tuple[synth.SynthPage, list[Region]]]:
    det, seg = ClassicalDetector(CFG.detect), FloodBubbleSegmenter(CFG.detect)
    out = []
    for page in [
        synth.basic_page("ja", 80, cache_dir),
        synth.basic_page("ko", 80, cache_dir),
        synth.variety_page("zh", 81, cache_dir),
        synth.texture_page(0, cache_dir),
        synth.adversarial_gutter_page(0, cache_dir),
        synth.two_block_bubble_page(0, cache_dir),
    ]:
        out.append((page, seg.segment(page.image, det.detect(page.image), page.name)))
    return out


def _union(regions: list[Region], h: int, w: int) -> np.ndarray:
    union = np.zeros((h, w), bool)
    for r in regions:
        if r.inpaint_mask is not None:
            union |= r.inpaint_mask.to_full(h, w)
    return union


@pytest.mark.parametrize("strategy", ["auto", "solid", "telea", "ns", "lama"])
def test_pixels_outside_mask_are_bit_identical(pages, strategy: str) -> None:  # type: ignore[no-untyped-def]
    for page, regions in pages:
        regs = [dataclasses.replace(r, flags=set(r.flags)) for r in regions]
        clean = _inpainter(strategy).inpaint_page(page.image, regs)
        h, w = page.image.shape[:2]
        changed = (clean != page.image).any(axis=2)
        assert not (changed & ~_union(regs, h, w)).any(), (page.name, strategy)
        assert clean.dtype == np.uint8 and clean.shape == page.image.shape


def test_bubble_outlines_untouched(pages) -> None:  # type: ignore[no-untyped-def]
    for page, regions in pages:
        clean = _inpainter().inpaint_page(page.image, regions)
        h, w = page.image.shape[:2]
        changed = (clean != page.image).any(axis=2)
        for g in page.regions:
            if g.bubble_mask is None or g.shape == "open":
                continue
            interior = g.bubble_mask.to_full(h, w)
            band = (
                cv2.dilate(interior.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            ) & ~interior
            assert not (changed & band).any(), (page.name, g.text)


def test_solid_fill_uniform_and_text_gone(pages) -> None:  # type: ignore[no-untyped-def]
    residual = total = 0
    for page, regions in pages:
        clean = _inpainter().inpaint_page(page.image, regions)
        h, w = page.image.shape[:2]
        gray = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY)
        for r in regions:
            if r.inpaint_method == "solid":
                assert float(gray[r.inpaint_mask.to_full(h, w)].std()) <= 2.0, r.id  # type: ignore[union-attr]
            if (
                r.bubble_mask is not None
                and r.inpaint_method in {"solid", "ns"}
                and Flag.LEAK_FALLBACK not in r.flags
            ):
                b = r.bubble_mask.bbox
                sub = gray[b.y0 : b.y1, b.x0 : b.x1]
                dev = np.abs(sub.astype(int) - cv2.medianBlur(sub, 15)) > 30
                inside = cv2.erode(r.bubble_mask.data.astype(np.uint8), np.ones((9, 9), np.uint8))
                total += 1
                residual += bool((dev & (inside > 0)).any())
    assert total >= 10 and residual / total <= 0.05, (residual, total)


def test_strategy_selection(pages) -> None:  # type: ignore[no-untyped-def]
    lama = FakeLama()
    methods: dict[str, set[str]] = {}
    for page, regions in pages:
        _inpainter(lama=lama).inpaint_page(page.image, regions)
        for r in regions:
            methods.setdefault(r.type.value, set()).add(str(r.inpaint_method))
    assert "solid" in methods["bubble"]
    assert "ns" in methods["bubble"] or "ns" in methods["free_text"]  # gradient fixtures
    assert lama.calls >= 1  # textured free text → LaMa


def test_lama_failure_falls_back_to_telea(cjk_ready: Path, cache_dir: Path) -> None:
    page = synth.texture_page(0, cache_dir)
    det, seg = ClassicalDetector(CFG.detect), FloodBubbleSegmenter(CFG.detect)
    regions = seg.segment(page.image, det.detect(page.image), page.name)
    lama = FakeLama(fail=True)
    clean = _inpainter(lama=lama).inpaint_page(page.image, regions)
    textured = [r for r in regions if r.type == RegionType.FREE_TEXT]
    assert lama.calls >= 1
    assert all(
        r.inpaint_method == "telea" and Flag.INPAINT_FALLBACK in r.flags
        for r in textured
        if r.inpaint_method != "solid"
    )
    assert (clean != page.image).any()
    no_lama = RegionInpainter(CFG.inpaint, None).inpaint_page(page.image, regions)
    assert {r.inpaint_method for r in textured} <= {"telea", "solid"}
    assert no_lama.shape == page.image.shape


def test_empty_mask_is_noop_and_border_regions_safe() -> None:
    page = np.full((120, 160, 3), 255, np.uint8)
    blank = Region(
        id="b",
        type=RegionType.BUBBLE,
        bbox=BBox(10, 10, 60, 60),
        text_mask=CropMask(BBox(10, 10, 60, 60), np.zeros((50, 50), bool)),
    )
    out = _inpainter().inpaint_page(page, [blank])
    assert np.array_equal(out, page) and blank.inpaint_method == "none"
    edge = page.copy()
    edge[0:12, 0:40] = 0  # text touching the top-left corner
    tm = np.zeros((12, 40), bool)
    tm[:, :] = True
    r = Region(
        id="e",
        type=RegionType.FREE_TEXT,
        bbox=BBox(0, 0, 40, 12),
        text_mask=CropMask(BBox(0, 0, 40, 12), tm),
        lines=[BBox(0, 0, 40, 12)],
    )
    out = _inpainter().inpaint_page(edge, [r])
    assert out.shape == edge.shape and r.inpaint_mask is not None
    assert r.inpaint_mask.bbox.x0 >= 0 and r.inpaint_mask.bbox.y0 >= 0


def test_grayscale_stays_grayscale(cjk_ready: Path, cache_dir: Path) -> None:
    page = synth.texture_page(1, cache_dir, "ko")
    gray = np.repeat(cv2.cvtColor(page.image, cv2.COLOR_RGB2GRAY)[..., None], 3, axis=2)
    det, seg = ClassicalDetector(CFG.detect), FloodBubbleSegmenter(CFG.detect)
    regions = seg.segment(gray, det.detect(gray), "g")
    for strategy in ("auto", "telea", "lama"):
        out = _inpainter(strategy).inpaint_page(gray, regions)
        assert np.array_equal(out[..., 0], out[..., 1]) and np.array_equal(out[..., 1], out[..., 2])


def test_residual_check_retries_then_flags() -> None:
    page = np.full((100, 200, 3), 255, np.uint8)
    page[40:60, 50:150] = 0
    tm = np.zeros((20, 100), bool)
    tm[:, :] = True
    region = Region(
        id="r",
        type=RegionType.FREE_TEXT,
        bbox=BBox(50, 40, 150, 60),
        text_mask=CropMask(BBox(50, 40, 150, 60), tm),
        lines=[BBox(50, 40, 150, 60)],
    )

    class AlwaysResidual:
        def has_residual(self, page: np.ndarray, region: Region) -> bool:
            return True

    cfg = dataclasses.replace(CFG.inpaint, residual_check=True, residual_max_passes=2)
    inp = RegionInpainter(cfg, None, AlwaysResidual())  # type: ignore[arg-type]
    inp.inpaint_page(page, [region])
    assert Flag.RESIDUAL_TEXT in region.flags
    real = RegionInpainter(cfg, None, ResidualChecker(ClassicalDetector(CFG.detect)))
    region2 = dataclasses.replace(region, flags=set())
    real.inpaint_page(page, [region2])
    assert Flag.RESIDUAL_TEXT not in region2.flags


def test_background_classifier() -> None:
    mask = np.zeros((60, 60), bool)
    mask[25:35, 25:35] = True
    flat = np.full((60, 60, 3), 240, np.uint8)
    assert classify_background(flat, mask, None, CFG.inpaint).kind == Background.UNIFORM
    yy, xx = np.mgrid[0:60, 0:60]
    ramp = np.stack([180 + xx, 200 - yy // 2, 230 - xx // 2], -1).astype(np.uint8)
    assert classify_background(ramp, mask, None, CFG.inpaint).kind == Background.GRADIENT
    tone = flat.copy()
    tone[::5, ::5] = 90
    tone[1::5, ::5] = 90
    assert classify_background(tone, mask, None, CFG.inpaint).kind == Background.TEXTURE


def test_mask_helpers() -> None:
    img = np.full((80, 120, 3), 255, np.uint8)
    img[20:60, 30:90] = 120  # "artwork"
    img[30:50, 44:76] = 255  # white halo
    img[36:44, 50:70] = 0  # glyph
    text = CropMask.from_full(np.all(img == 0, axis=2))
    assert text is not None
    halo = halo_mask(img, text, glyph=10)
    assert halo is not None and halo.area > text.area
    assert stroke_width(text.data) >= 6
    bubble = np.full((60, 60, 3), 255, np.uint8)
    bubble[20:30, 20:30] = 0
    bubble[32, 24] = 200  # faint anti-aliased speck 3 px below the glyph
    glyph = CropMask.from_full(np.all(bubble == 0, axis=2))
    assert glyph is not None
    stray = stray_ink(bubble, glyph, glyph=10)
    assert stray is not None and stray.window(BBox(24, 32, 25, 33))[0, 0]
    region = Region(id="m", type=RegionType.BUBBLE, bbox=BBox(20, 20, 30, 30), text_mask=glyph)
    m = build_mask(bubble, region, CFG.inpaint)
    assert m is not None and m.window(BBox(24, 32, 25, 33))[0, 0]


def test_allowed_zone_keeps_outline_at_mask_extremes() -> None:
    from manga_ar.detect.geometry import ellipse_mask
    from manga_ar.inpaint.mask import allowed_zone

    cfg = load_config(environ={}).inpaint
    box = BBox(10, 10, 110, 90)
    region = Region(
        id="z",
        type=RegionType.BUBBLE,
        bbox=box,
        bubble_mask=CropMask(box, ellipse_mask((80, 100), BBox(0, 0, 100, 80))),
    )
    zone = allowed_zone(region, (120, 120), cfg, 2).data
    e = cfg.allowed_erode
    assert not zone[:e].any() and not zone[-e:].any()
    assert not zone[:, :e].any() and not zone[:, -e:].any()
