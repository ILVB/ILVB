"""SP-B: manga-ocr on synthetic vertical + horizontal Japanese.

manga-ocr downloads `kha-white/manga-ocr-base` from the Hugging Face Hub on first use.
"""

from __future__ import annotations

import time

from PIL import Image

from _common import OUT, text_image


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    samples = {
        "horizontal": text_image("今日はいい天気ですね", "ja"),
        "vertical": text_image("今日は\nいい天気", "ja", vertical=True),
    }
    try:
        from manga_ocr import MangaOcr

        t0 = time.perf_counter()
        mocr = MangaOcr()
        print(f"model loaded in {time.perf_counter() - t0:.1f}s")
    except Exception as exc:
        print(f"VERDICT: FAIL (model load) {type(exc).__name__}: {str(exc)[:300]}")
        return
    for name, arr in samples.items():
        print(name, repr(mocr(Image.fromarray(arr))))
    print("VERDICT: PASS")


if __name__ == "__main__":
    main()
