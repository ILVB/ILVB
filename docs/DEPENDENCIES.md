# Dependencies

All versions are pinned in `uv.lock` (resolved for Python 3.10–3.12, all platforms).
Install exactly: `uv sync --locked --extra dev --extra rapid --extra ocr --extra gui`
(add `lama`, `manga`, `offline-mt`, `ctd`, `paddle`, `gpu` as needed).
pip users: `pip install -e ".[rapid,ocr,gui]"`, then run `manga-arabic doctor` to catch
duplicate OpenCV distributions (see "OpenCV" below).

## Core (always installed)
| Package | Pinned | Licence | Purpose |
|---|---|---|---|
| numpy | 2.4.6 (py3.11) | BSD-3 | arrays |
| opencv-python-headless | 5.0.0.93 | Apache-2.0 | CV ops, inpainting (Telea/NS) |
| pillow | 12.3.0 | MIT-CMU (HPND) | decoding/encoding, text rendering (BASIC + RAQM) |
| fonttools | 4.66.1 | MIT | cmap/axes inspection (glyph coverage) |
| pyyaml | 6.0.3 | MIT | configuration |
| platformdirs | 4.12.2 | MIT | default cache directory |
| huggingface_hub | 1.33.0 | Apache-2.0 | optional model downloads (manga-ocr, local MT) |
| requests | 2.34.2 | Apache-2.0 | downloads, reachability, LibreTranslate |
| arabic-reshaper | 3.0.1 | MIT | Arabic contextual shaping (BASIC path) |
| python-bidi | 0.6.11 | LGPL-3.0 | Unicode bidi levels (see licence note) |
| deep-translator | 1.11.4 | Apache-2.0 | Google web endpoint / MyMemory / LibreTranslate |
| tqdm | 4.70.1 | MIT + MPL-2.0 | CLI progress |

## Extras
| Extra | Packages (pinned) | Licence | Why |
|---|---|---|---|
| `rapid` | rapidocr_onnxruntime 1.4.4, onnxruntime 1.30.0 | Apache-2.0, MIT | PP-OCR ONNX models **bundled in the wheel** (zh OCR + DB text detector) without Paddle or model downloads. This is the substitute the prompt allows when PaddlePaddle models are unavailable (DECISIONS SP-C). |
| `ocr` | easyocr 1.7.2, torch 2.14.0, torchvision 0.29.0 | Apache-2.0, BSD-3 | ja/ko/zh OCR + CRAFT detector; weights from GitHub releases |
| `lama` | torch | BSD-3 | LaMa big-lama TorchScript inpainting (loaded directly; the `simple-lama-inpainting` wrapper was dropped because it pulls opencv-python + fire) |
| `manga` | manga-ocr 0.1.16 | Apache-2.0 | Japanese OCR (HF weights) |
| `offline-mt` | transformers 5.17.0, sentencepiece 0.2.2, sacremoses | Apache-2.0, Apache-2.0, MIT | Marian/M2M100 local translation |
| `ctd` | onnxruntime | MIT | comic-text-detector ONNX runtime (weights GPL-3.0, opt-in) |
| `paddle` | paddleocr, paddlepaddle | Apache-2.0 | optional; models blocked on the build host (W-002) |
| `gpu` | onnxruntime-gpu | MIT | optional CUDA for ONNX models |
| `gui` | gradio 6.29.0, gradio_client 2.7.1 | Apache-2.0 | local web GUI |
| `dev` | pytest 9.1.1, pytest-cov, hypothesis 6.168.3, ruff 0.16.9, mypy 2.3.1, pip-licenses, types-* | MIT/Apache/MPL | quality gates |

## Additions beyond the prompt's closed list
- **rapidocr_onnxruntime / onnxruntime.** The prompt names these as the substitute when
  PaddlePaddle cannot be used. Size ≈ 15 MB + 15 MB. No allowed alternative ships PP-OCR
  weights without a blocked download.
- **sacremoses.** Optional tokenizer dependency of Marian models in transformers (MIT, < 1 MB).
- Transitive only (not imported by MangaAR): shapely, pyclipper, scikit-image (EasyOCR),
  fire (a dependency of several OCR packages).

## OpenCV (exactly one distribution)
easyocr requires `opencv-python-headless`. `rapidocr_onnxruntime` and PaddleOCR require
`opencv-python`/`opencv-contrib-python`. All of them import the same `cv2` module, and two
installed distributions corrupt each other. `pyproject.toml` therefore contains a uv
override that drops `opencv-python` and `opencv-contrib-python`. The lock file shows the
edges as `marker = "sys_platform == 'never'"`. pip cannot express this, so pip users run:
`pip uninstall -y opencv-python opencv-contrib-python && pip install --force-reinstall
opencv-python-headless`. `manga-arabic doctor` fails loudly while duplicates exist.

## API changes verified against installed versions (Phase 0 spikes)
- python-bidi 0.6: the Rust `get_display` does no L4 mirroring, so the shim mirrors itself.
- deep-translator 1.11.4: no timeout parameter (executor deadline); Google `< 5000`,
  MyMemory `< 500` characters.
- Pillow 12: `getbbox`/`getlength`/`textbbox` only; `ImageFont.Layout.BASIC|RAQM`.
- OpenCV 5.0: `cv2.inpaint`, `MSER_create` and `imdecode` all verified present.
- Gradio 6.29: `Blocks.queue(default_concurrency_limit=1)` and `launch(server_name=…)`.
- PaddleOCR 3.7: `PaddleOCR(lang=…).predict()` (the 2.x `.ocr()` is kept as a fallback).

## torch and NVIDIA libraries (licence flag)
On Linux x86_64, the PyPI `torch` wheel pinned in `uv.lock` (2.14.0) depends on the CUDA
runtime (`nvidia-*`, `cuda-toolkit`). These libraries are proprietary but redistributable
under NVIDIA's EULA. Only the torch-based extras pull them: `ocr`, `lama`, `manga` and
`offline-mt`.

For a fully open-source environment, install the CPU build first, then MangaAR:
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.
Re-locking uv against that index needs download.pytorch.org, which this build host
blocks (D-043). Windows and macOS PyPI torch wheels contain no NVIDIA libraries.
