"""SP-D: text-detector candidates vs. a classical baseline on a synthetic page.

Candidates: RapidOCR PP-OCR DB detector (bundled, Apache-2.0), EasyOCR CRAFT (GitHub
weights), comic-text-detector ONNX (GPL-3.0 weights, GitHub release), classical CV.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from _common import CACHE, FONT_FILES, OUT

CTD_URL = (
    "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/"
    "comictextdetector.pt.onnx"
)


def synth_page() -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """800x1100 page: 2 panels, 3 ellipse bubbles with vertical JA text, 1 caption."""
    img = Image.new("RGB", (800, 1100), "white")
    d = ImageDraw.Draw(img)
    d.rectangle((20, 20, 780, 540), outline="black", width=4)
    d.rectangle((20, 560, 780, 1080), outline="black", width=4)
    # screentone-ish artwork
    for y in range(600, 1060, 6):
        for x in range(420, 760, 6):
            d.ellipse((x, y, x + 2, y + 2), fill=(90, 90, 90))
    font = ImageFont.truetype(str(FONT_FILES["ja"]), 30)
    boxes = []
    for cx, cy, cols in (
        (600, 250, ["今日は", "晴れ"]),
        (220, 300, ["本当に", "行くの"]),
        (230, 800, ["待って", "くれ"]),
    ):
        d.ellipse((cx - 110, cy - 150, cx + 110, cy + 150), fill="white", outline="black", width=3)
        x0 = cx + 25
        ys = cy - 60
        xs = []
        for ci, col in enumerate(cols):
            x = x0 - ci * 44
            for ri, ch in enumerate(col):
                d.text((x - 15, ys + ri * 34), ch, font=font, fill="black")
            xs.append(x)
        boxes.append((min(xs) - 17, ys - 2, max(xs) + 17, ys + 3 * 34 + 2))
    d.rectangle((440, 580, 760, 640), fill="white", outline="black", width=2)
    d.text((455, 590), "その日の夜", font=font, fill="black")
    boxes.append((455, 588, 610, 628))
    return np.asarray(img).copy(), boxes


def classical(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15
    )
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    mask = np.zeros_like(binary)
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if 6 <= w <= 60 and 6 <= h <= 60 and a > 15:
            mask[labels == i] = 255
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    return [tuple(int(v) for v in (s[0], s[1], s[0] + s[2], s[1] + s[3])) for s in stats[1:]]


def rapid_det(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR()
    res, _ = engine(img, use_det=True, use_cls=False, use_rec=False)
    out = []
    for quad in res or []:
        q = np.asarray(quad)
        out.append((int(q[:, 0].min()), int(q[:, 1].min()), int(q[:, 0].max()), int(q[:, 1].max())))
    return out


def craft_det(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    import easyocr

    reader = easyocr.Reader(
        ["ja", "en"],
        gpu=False,
        verbose=False,
        model_storage_directory=str(CACHE / "models/easyocr"),
    )
    horizontal, free = reader.detect(img)
    boxes = [tuple(int(v) for v in (b[0], b[2], b[1], b[3])) for b in horizontal[0]]
    for poly in free[0]:
        p = np.asarray(poly)
        boxes.append(
            (int(p[:, 0].min()), int(p[:, 1].min()), int(p[:, 0].max()), int(p[:, 1].max()))
        )
    return boxes


def ctd_det(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    import onnxruntime as ort
    import requests

    path = CACHE / "models/ctd/comictextdetector.pt.onnx"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(CTD_URL, timeout=(5, 120))
        r.raise_for_status()
        path.write_bytes(r.content)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    print(
        "ctd inputs",
        [(i.name, i.shape) for i in sess.get_inputs()],
        "outputs",
        [(o.name, o.shape) for o in sess.get_outputs()],
    )
    h, w = img.shape[:2]
    size = 1024
    scale = size / max(h, w)
    resized = cv2.resize(img, (int(w * scale), int(h * scale)))
    canvas = np.zeros((size, size, 3), np.uint8)
    canvas[: resized.shape[0], : resized.shape[1]] = resized
    blob = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    outs = sess.run(None, {sess.get_inputs()[0].name: blob})
    blks, seg, _lines = outs
    print("ctd output shapes", [o.shape for o in outs])
    preds = blks[0]
    keep = preds[preds[:, 4] > 0.4]
    boxes = []
    for x, y, bw, bh, *_ in keep:
        boxes.append(
            (
                int((x - bw / 2) / scale),
                int((y - bh / 2) / scale),
                int((x + bw / 2) / scale),
                int((y + bh / 2) / scale),
            )
        )
    idx = cv2.dnn.NMSBoxes(
        [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes], [1.0] * len(boxes), 0.4, 0.45
    )
    return [boxes[i] for i in np.asarray(idx).flatten()] if len(boxes) else []


def iou(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    page, truth = synth_page()
    Image.fromarray(page).save(OUT / "sp_d_page.png")
    for name, fn in (
        ("classical", classical),
        ("rapid_db", rapid_det),
        ("craft", craft_det),
        ("ctd", ctd_det),
    ):
        try:
            t0 = time.perf_counter()
            boxes = fn(page)
            dt = time.perf_counter() - t0
        except Exception as exc:
            print(f"{name}: FAILED {type(exc).__name__}: {str(exc)[:200]}")
            continue
        matched = sum(1 for t in truth if any(iou(t, b) >= 0.3 for b in boxes))
        vis = page.copy()
        for b in boxes:
            cv2.rectangle(vis, b[:2], b[2:], (255, 0, 0), 2)
        for t in truth:
            cv2.rectangle(vis, t[:2], t[2:], (0, 180, 0), 1)
        Image.fromarray(vis).save(OUT / f"sp_d_{name}.png")
        print(f"{name}: {len(boxes)} boxes, truth matched@IoU0.3 {matched}/{len(truth)}, {dt:.2f}s")


if __name__ == "__main__":
    main()
