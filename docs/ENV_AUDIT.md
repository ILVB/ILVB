# Environment Audit (Phase 0.1)

Audited 2026-09-30 on the build host (cloud container).

| Item | Finding |
|---|---|
| OS / arch | Ubuntu 24.04.4 LTS, Linux 6.18 x86_64 |
| CPU | 4 × Intel Xeon @ 2.10 GHz |
| RAM | 15.7 GiB, no swap |
| GPU | none (no `nvidia-smi`; `torch.cuda.is_available() == False`; no MPS) |
| Free disk | 30 GB at start (22 GB after the ML stack + uv cache); requirement ≥ 10 GB met |
| Python interpreters | 3.10, 3.11.15 (default `python3`), 3.12, 3.13 |
| Chosen interpreter | 3.11.15 in `.venv/` created with `uv venv --python 3.11` |
| Package tools | uv 0.8.17, pip (system) — only uv/venv used, no global installs |
| Compiler toolchain | gcc 13.3, clang 18.1, make |
| System fonts of note | DejaVu Sans, WenQuanYi Zen Hei (CJK+Hangul), IPAGothic, Noto Color Emoji |
| Pillow | 12.3.0, FreeType 2.14.3, **libraqm 0.10.5 available** (HarfBuzz 14.2.1, FriBiDi) |
| OpenCV | 5.0.0 (`opencv-python-headless`); duplicate `opencv-python` pulled by simple-lama/rapidocr was removed |
| torch | 2.14.0+cu130 (PyPI build, CPU execution) |

## Network reachability (via the session egress proxy)

| Host | Purpose | Result |
|---|---|---|
| pypi.org / files.pythonhosted.org | packages | ✅ 200 |
| github.com release assets (objects/release-assets.githubusercontent.com) | EasyOCR, LaMa, comic-text-detector weights | ✅ 206 (range OK) |
| raw.githubusercontent.com | fonts (google/fonts, notofonts/noto-cjk) | ✅ 200 |
| api.github.com, codeload.github.com | — | ❌ 403 |
| huggingface.co, cdn-lfs.huggingface.co | manga-ocr, Marian/M2M100/NLLB, HF detector | ❌ 403 (policy) |
| translate.google.com | deep-translator Google | ❌ 403 (policy) |
| api.mymemory.translated.net | deep-translator MyMemory | ❌ 403 (policy) |
| libretranslate.com | public LibreTranslate | ❌ blocked |
| fonts.google.com | fonts | ❌ 403 (google/fonts GitHub mirror used instead) |
| paddle-model-ecology.bj.bcebos.com, paddleocr.bj.bcebos.com, aistudio.baidu.com, modelscope.cn | PaddleOCR models | ❌ blocked |
| argos-net.com, ipfs gateways | Argos Translate models | ❌ blocked |
| object.pouta.csc.fi | OPUS-MT original models | ❌ blocked |
| dl.fbaipublicfiles.com | M2M100 original checkpoints | ❌ blocked |
| download.pytorch.org | CPU-only torch wheels | ❌ blocked (PyPI build used) |

## Consequences
- **Usable models here:** EasyOCR (ja/ko/ch_sim/ch_tra/ar + CRAFT detector), RapidOCR PP-OCR
  (bundled in the wheel, zh), LaMa big-lama TorchScript, comic-text-detector ONNX (GPL-3.0).
- **Not usable here:** manga-ocr, any Hugging Face model (local MT, HF bubble detector),
  PaddleOCR models, every free online translation endpoint.
- These paths are still implemented and unit-tested with fakes; their live gates carry
  explicit waivers in `DECISIONS.md` with the command to run them on an unrestricted host.
- Gradio's server makes no outbound calls at launch; `gradio_client` pings huggingface.co
  once when a client is created (tests set `HF_HUB_OFFLINE=1`).
