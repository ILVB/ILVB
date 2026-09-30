"""SP-C: OCR engines for Korean / Chinese / Japanese on synthetic images.

- RapidOCR (PP-OCR ONNX; Chinese models bundled in the wheel, no download).
- EasyOCR (models from GitHub releases on first use).
PaddleOCR itself is probed separately in an isolated venv (see sp_c_paddle_probe.py).
"""

from __future__ import annotations

import os
import time

from _common import CACHE, OUT, text_image

SAMPLES = {
    "zh": "今天天气很好",
    "ko": "오늘은 날씨가 좋네요",
    "ja": "今日はいい天気ですね",
}


def run_rapid() -> None:
    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR()
    for lang, text in SAMPLES.items():
        t0 = time.perf_counter()
        result, _ = engine(text_image(text, lang))
        got = "".join(r[1] for r in result) if result else ""
        print(f"rapid {lang}: {got!r} (truth {text!r}) {time.perf_counter() - t0:.2f}s")
    vert = text_image("今天天气\n很好", "zh", vertical=True)
    result, _ = engine(vert)
    print("rapid zh vertical raw:", [(r[1], round(float(r[2]), 2)) for r in (result or [])])


def run_easyocr() -> None:
    import easyocr

    model_dir = CACHE / "models" / "easyocr"
    model_dir.mkdir(parents=True, exist_ok=True)
    for lang, codes in {"ko": ["ko", "en"], "zh": ["ch_sim", "en"], "ja": ["ja", "en"]}.items():
        t0 = time.perf_counter()
        reader = easyocr.Reader(
            codes,
            gpu=False,
            model_storage_directory=str(model_dir),
            download_enabled=True,
            verbose=False,
        )
        t1 = time.perf_counter()
        res = reader.readtext(text_image(SAMPLES[lang], lang), detail=1, paragraph=False)
        got = "".join(r[1] for r in res)
        print(
            f"easyocr {lang}: {got!r} conf={[round(float(r[2]), 2) for r in res]} "
            f"load={t1 - t0:.1f}s infer={time.perf_counter() - t1:.2f}s"
        )


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    for fn in (run_rapid, run_easyocr):
        try:
            fn()
        except Exception as exc:
            print(f"{fn.__name__} FAILED: {type(exc).__name__}: {str(exc)[:300]}")
