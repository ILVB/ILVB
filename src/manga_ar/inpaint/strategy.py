"""Per-region inpainting strategy and exact compositing (S4).

Background classes: uniform → solid fill; smooth gradient → Navier-Stokes; textured →
LaMa (context crop) → Telea fallback. Only mask pixels are ever written, so everything
outside the (dilated) mask stays bit-identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from manga_ar.config import InpaintConfig
from manga_ar.detect.geometry import disk
from manga_ar.errors import InpaintError, ModelUnavailableError
from manga_ar.inpaint.base import BoolArray, Inpainter, RgbArray
from manga_ar.inpaint.mask import allowed_zone, build_mask, stroke_width
from manga_ar.inpaint.opencv_inpaint import OpenCvInpainter
from manga_ar.inpaint.residual import ResidualChecker
from manga_ar.inpaint.solid_fill import SolidFill
from manga_ar.logging_setup import get_logger
from manga_ar.schemas import CropMask, Flag, Region, RegionType

log = get_logger(__name__)


class Background(str, Enum):
    UNIFORM = "uniform"
    GRADIENT = "gradient"
    TEXTURE = "texture"


@dataclass
class BackgroundStats:
    kind: Background
    std: float
    residual_std: float


def _robust_std(values: np.ndarray) -> float:
    """1.4826 × median absolute deviation: a std-dev estimate immune to a few outliers
    (anti-aliasing or stray pixels next to the mask)."""
    med = np.median(values)
    return float(1.4826 * np.median(np.abs(values - med)))


def classify_background(
    image: RgbArray, mask: BoolArray, zone: BoolArray | None, cfg: InpaintConfig
) -> BackgroundStats:
    """Classify the ring around ``mask`` as uniform, smooth gradient or texture.

    Robust per-channel spread (1.4826·MAD) ignores a few stray pixels, but sparse texture
    (screentone dots, hatching: 10–30 % of pixels) also has MAD ≈ 0, so the share of
    pixels deviating > 20 levels is checked too (calibrated: bubbles ≤ 0.6 %, textures
    ≥ 10 %; cut-off 3 %). Gradients are judged on the residual of a per-channel plane.
    """
    m = mask.astype(np.uint8)
    ring = (cv2.dilate(m, disk(10)) > 0) & ~(cv2.dilate(m, disk(2)) > 0)
    if zone is not None:
        ring &= zone
    ys, xs = np.nonzero(ring)
    if ys.size < 12:
        return BackgroundStats(Background.TEXTURE, 255.0, 255.0)
    rgb = image[ys, xs].astype(np.float64)
    spread = max(_robust_std(rgb[:, c]) for c in range(3))
    outliers = float((np.abs(rgb - np.median(rgb, axis=0)).max(axis=1) > 20).mean())
    if spread <= cfg.uniform_std and outliers <= 0.03:
        return BackgroundStats(Background.UNIFORM, spread, spread)
    design = np.column_stack([xs, ys, np.ones_like(xs)]).astype(np.float64)
    residuals = np.empty_like(rgb)
    for c in range(3):
        coef, *_ = np.linalg.lstsq(design, rgb[:, c], rcond=None)
        residuals[:, c] = rgb[:, c] - design @ coef
    residual = max(_robust_std(residuals[:, c]) for c in range(3))
    res_outliers = float((np.abs(residuals).max(axis=1) > 20).mean())
    if residual <= cfg.gradient_residual_std and res_outliers <= 0.03:
        return BackgroundStats(Background.GRADIENT, spread, residual)
    return BackgroundStats(Background.TEXTURE, spread, residual)


def _is_gray(img: RgbArray) -> bool:
    return bool(
        np.array_equal(img[..., 0], img[..., 1]) and np.array_equal(img[..., 1], img[..., 2])
    )


class RegionInpainter:
    """Cleans every eligible region of a page."""

    def __init__(
        self,
        cfg: InpaintConfig,
        lama: Inpainter | None = None,
        residual: ResidualChecker | None = None,
    ) -> None:
        self.cfg = cfg
        self.lama = lama
        self.residual = residual
        self._lama_broken = False

    # ---------------------------------------------------------------- page
    def inpaint_page(self, page: RgbArray, regions: list[Region]) -> RgbArray:
        out = page.copy()
        for region in regions:
            if not self.eligible(region):
                continue
            self.inpaint_region(out, region)
        return out

    @staticmethod
    def eligible(region: Region) -> bool:
        if region.type == RegionType.SFX or region.override.skip:
            return False
        return Flag.PASS_THROUGH not in region.flags

    # -------------------------------------------------------------- region
    def inpaint_region(self, page: RgbArray, region: Region) -> str:
        """Clean ``region`` in place on ``page``; returns the method used."""
        method = "none"
        for attempt in range(self.cfg.residual_max_passes + 1 if self.cfg.residual_check else 1):
            method = self._clean(page, region, extra_dilate=3 * attempt)
            if method == "none" or self.residual is None or not self.cfg.residual_check:
                break
            if not self.residual.has_residual(page, region):
                region.flags.discard(Flag.RESIDUAL_TEXT)
                break
            region.flag(Flag.RESIDUAL_TEXT)
            log.info("%s: residual text after %s (pass %d)", region.id, method, attempt + 1)
        region.inpaint_method = method
        return method

    def _clean(self, page: RgbArray, region: Region, extra_dilate: int) -> str:
        h, w = page.shape[:2]
        mask = build_mask(page, region, self.cfg, extra_dilate)
        if mask is None:
            region.inpaint_mask = None
            return "none"
        region.inpaint_mask = mask
        sw = stroke_width(region.text_mask.data) if region.text_mask is not None else 3.0
        zone_full = allowed_zone(region, (h, w), self.cfg, int(0.5 * sw + 1))
        chain = self._chain(page, region, mask, zone_full)
        context = self.cfg.lama_context if chain[0] == "lama" else 24
        box = mask.bbox.expand(context).clip(w, h)
        crop = page[box.y0 : box.y1, box.x0 : box.x1].copy()
        m = mask.window(box)
        zone = zone_full.window(box)
        for i, name in enumerate(chain):
            try:
                result = self._run(name, crop, m, zone)
            except (InpaintError, ModelUnavailableError) as exc:
                if name == "lama":
                    self._lama_broken = isinstance(exc, ModelUnavailableError)
                log.warning("%s: %s inpainting failed (%s); falling back", region.id, name, exc)
                region.flag(Flag.INPAINT_FALLBACK)
                continue
            if _is_gray(crop):
                result = np.repeat(result.mean(axis=2, keepdims=True), 3, axis=2)
                result = np.rint(result).astype(np.uint8)
            view = page[box.y0 : box.y1, box.x0 : box.x1]
            view[m] = result[m]  # exact composite: only mask pixels change
            if i > 0:
                region.flag(Flag.INPAINT_FALLBACK)
            return name
        raise InpaintError(f"{region.id}: every inpainting method failed")

    def _chain(self, page: RgbArray, region: Region, mask: CropMask, zone: CropMask) -> list[str]:
        forced = self.cfg.strategy
        lama_ok = self.lama is not None and self.cfg.use_lama and not self._lama_broken
        if forced != "auto":
            base = {
                "solid": ["solid"],
                "telea": ["telea"],
                "ns": ["ns", "telea"],
                "lama": ["lama", "telea"],
            }[forced]
            return [
                n for n in base if n != "lama" or (self.lama is not None and not self._lama_broken)
            ] or ["telea"]
        h, w = page.shape[:2]
        box = mask.bbox.expand(12).clip(w, h)
        stats = classify_background(
            page[box.y0 : box.y1, box.x0 : box.x1], mask.window(box), zone.window(box), self.cfg
        )
        if stats.kind == Background.UNIFORM:
            return ["solid"]
        if stats.kind == Background.GRADIENT:
            return ["ns", "telea"]
        return ["lama", "telea"] if lama_ok else ["telea"]

    def _run(self, name: str, crop: RgbArray, mask: BoolArray, zone: BoolArray) -> RgbArray:
        if name == "solid":
            return SolidFill(zone=zone, feather=self.cfg.feather_px).inpaint(crop, mask)
        if name in {"telea", "ns"}:
            return OpenCvInpainter(name).inpaint(crop, mask)
        if name == "lama":
            if self.lama is None:
                raise ModelUnavailableError("LaMa not configured")
            return self.lama.inpaint(crop, mask)
        raise InpaintError(f"unknown inpainting method {name!r}")
