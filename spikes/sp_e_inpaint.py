"""SP-E: solid fill vs. Telea vs. Navier-Stokes vs. LaMa on a gradient bubble.

LaMa: TorchScript big-lama from the IOPaint model release (GitHub), loaded directly with
torch.jit.load (no simple-lama wrapper; it would pull opencv-python and fire).
"""

from __future__ import annotations

import hashlib
import time

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

from _common import CACHE, FONT_FILES, OUT

LAMA_URL = "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"


def scene() -> tuple[np.ndarray, np.ndarray]:
    h, w = 360, 480
    yy, xx = np.mgrid[0:h, 0:w]
    grad = np.stack([180 + 60 * xx / w, 200 - 50 * yy / h, 230 - 30 * xx / w], -1).astype(np.uint8)
    img = Image.fromarray(grad)
    d = ImageDraw.Draw(img)
    d.ellipse((40, 30, 440, 330), outline="black", width=4)
    font = ImageFont.truetype(str(FONT_FILES["ja"]), 44)
    d.text((120, 140), "グラデーション", font=font, fill="black")
    arr = np.asarray(img).copy()
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    text = np.zeros((h, w), np.uint8)
    text[120:210, 100:400] = (gray[120:210, 100:400] < 100).astype(np.uint8) * 255
    mask = cv2.dilate(text, np.ones((7, 7), np.uint8))
    return arr, mask


def lama(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    import torch

    path = CACHE / "models/lama/big-lama.pt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(LAMA_URL, stream=True, timeout=(5, 300)) as r:
            r.raise_for_status()
            with path.open("wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
    print("big-lama sha256", hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size)
    t0 = time.perf_counter()
    model = torch.jit.load(str(path), map_location="cpu").eval()
    print(f"lama load {time.perf_counter() - t0:.1f}s")
    h, w = img.shape[:2]
    ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
    im = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="symmetric")
    mk = np.pad(mask, ((0, ph), (0, pw)), mode="symmetric")
    ti = torch.from_numpy(im).permute(2, 0, 1)[None].float() / 255
    tm = (torch.from_numpy(mk)[None, None].float() > 127).float()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model(ti, tm)
    print(f"lama infer {time.perf_counter() - t0:.2f}s")
    res = (out[0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)[:h, :w]
    return np.where(mask[..., None] > 0, res, img)


def residual_std(img: np.ndarray, mask: np.ndarray, ref: np.ndarray) -> float:
    m = mask > 0
    return float(np.abs(img[m].astype(int) - ref[m].astype(int)).mean())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    img, mask = scene()
    # reference: same gradient without text (the "ideal" fill)
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    ref = np.stack([180 + 60 * xx / w, 200 - 50 * yy / h, 230 - 30 * xx / w], -1).astype(np.uint8)
    results = {}
    ring = cv2.dilate(mask, np.ones((9, 9), np.uint8)) & ~mask
    med = np.median(img[ring > 0], axis=0).astype(np.uint8)
    solid = img.copy()
    solid[mask > 0] = med
    results["solid"] = solid
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    results["telea"] = cv2.cvtColor(cv2.inpaint(bgr, mask, 5, cv2.INPAINT_TELEA), cv2.COLOR_BGR2RGB)
    results["ns"] = cv2.cvtColor(cv2.inpaint(bgr, mask, 5, cv2.INPAINT_NS), cv2.COLOR_BGR2RGB)
    try:
        results["lama"] = lama(img, mask)
    except Exception as exc:
        print(f"lama FAILED {type(exc).__name__}: {str(exc)[:200]}")
    tiles = [img] + list(results.values())
    Image.fromarray(np.concatenate(tiles, axis=1)).save(OUT / "sp_e_inpaint.png")
    for k, v in results.items():
        outside_same = bool(np.array_equal(v[mask == 0], img[mask == 0]))
        print(
            f"{k:6s} mean|err| vs clean gradient inside mask = {residual_std(v, mask, ref):5.2f}"
            f"  outside-mask identical={outside_same}"
        )


if __name__ == "__main__":
    main()
