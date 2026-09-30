"""Inpainting metrics: background integrity, masked-region SSIM/PSNR, residual text (G-INP)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

U8 = npt.NDArray[np.uint8]
Bool = npt.NDArray[np.bool_]
Box = tuple[int, int, int, int]
PSNR_CAP = 100.0


def _dilate(mask: Bool, radius: int) -> Bool:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return np.asarray(cv2.dilate(mask.astype(np.uint8), k) > 0, dtype=np.bool_)


def changed_outside(original: U8, erased: U8, declared: Bool, ring_px: int = 2) -> int:
    """G-INP-1: pixels that differ outside the declared erase mask plus a feather ring."""
    allowed = _dilate(declared, ring_px) if declared.any() else declared
    diff = np.any(original != erased, axis=2)
    return int((diff & ~allowed).sum())


def crop_box(mask: Bool, margin: int) -> Box:
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    return (max(0, int(xs.min()) - margin), max(0, int(ys.min()) - margin),
            min(w, int(xs.max()) + 1 + margin), min(h, int(ys.max()) + 1 + margin))  # fmt: skip


def ssim_psnr(erased: U8, clean: U8, box: Box) -> tuple[float, float]:
    """G-INP-2: SSIM and PSNR of the erased crop against the ground-truth clean crop."""
    x0, y0, x1, y1 = box
    a, b = erased[y0:y1, x0:x1], clean[y0:y1, x0:x1]
    win = min(7, (min(a.shape[:2]) // 2) * 2 + 1)
    if win < 3:
        return (1.0 if np.array_equal(a, b) else 0.0), PSNR_CAP
    ssim = float(structural_similarity(a, b, channel_axis=2, data_range=255, win_size=win))
    psnr = (
        PSNR_CAP if np.array_equal(a, b) else float(peak_signal_noise_ratio(b, a, data_range=255))
    )
    return ssim, min(psnr, PSNR_CAP)


def pixel_residual(erased: U8, clean: U8, text_mask: Bool, tol: int = 48) -> bool:
    """Ground-truth residual check: lettering-like deviations left inside the text area."""
    area = _dilate(text_mask, 2)
    dev = np.abs(erased.astype(np.int16) - clean.astype(np.int16)).max(axis=2) > tol
    hits = int((dev & area).sum())
    return hits > max(8, int(0.02 * int(text_mask.sum())))


@lru_cache(maxsize=1)
def _probe() -> Any:
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def probe_residual(erased: U8, box: Box) -> bool:
    """Detector re-run (G-INP-3): the PP-OCR DB text detector finds text in the region."""
    x0, y0, x1, y1 = box
    crop = np.ascontiguousarray(erased[y0:y1, x0:x1])
    if min(crop.shape[:2]) < 8:
        return False
    boxes, _ = _probe()(crop, use_det=True, use_cls=False, use_rec=False)
    return bool(boxes)
