"""SP-C (part 2): PaddleOCR probe, run in an isolated venv (.cache/venv-paddle).

Records the installed major version, the constructor/predict API, and whether model
download works from this host. Output normalisation adapter draft: see
src/manga_ar/ocr/paddle_engine.py.
"""

from __future__ import annotations

import inspect
import sys

import numpy as np


def main() -> None:
    import paddleocr

    print("paddleocr", paddleocr.__version__, "python", sys.version.split()[0])
    from paddleocr import PaddleOCR

    params = list(inspect.signature(PaddleOCR.__init__).parameters)
    print("PaddleOCR.__init__ params:", params[:25])
    print("has predict:", hasattr(PaddleOCR, "predict"), "has ocr:", hasattr(PaddleOCR, "ocr"))
    for lang in ("korean", "ch", "japan"):
        try:
            engine = PaddleOCR(
                lang=lang,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
            img = np.full((64, 256, 3), 255, np.uint8)
            res = engine.predict(img)
            print(lang, "OK", type(res))
        except Exception as exc:
            print(f"{lang}: FAILED {type(exc).__name__}: {str(exc)[:240]}")


if __name__ == "__main__":
    main()
