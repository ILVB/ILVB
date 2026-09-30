# MangaAR — Manga/Manhwa/Manhua → Arabic

MangaAR translates comic pages into Arabic on your own computer. It finds the text,
reads it (OCR), removes it cleanly, translates it and letters the Arabic back into the
bubbles: connected, right-to-left and sized to fit each bubble. It is a command-line
tool (`manga-arabic`) with a local web GUI.

- **Free and open source.** It needs no API keys, no accounts and no paid services,
  and sends no telemetry.
- **Runs on a CPU.** A GPU is used automatically when present.
- **Works offline.** Translation needs a local model or a translation memory; see
  [Offline use](#offline-use).
- **Inputs:** images (PNG, JPEG, WebP, BMP, TIFF), folders (recursive) and CBZ/ZIP
  archives, including long webtoon strips.
- **Resilient batches:** a broken page or region never stops a batch. Every page gets a
  sidecar JSON so you can fix a translation and re-render the page in seconds.

> Translate only material you have the right to translate. See
> [Responsible use](#responsible-use).

## Contents
[Install](#install) · [Quick start](#quick-start) · [CLI](#command-line) ·
[GUI](#gui) · [Presets](#presets) · [Translation & offline use](#translation) ·
[Models & downloads](#models-and-downloads) · [Troubleshooting](#troubleshooting) ·
[Responsible use](#responsible-use) · [Licences](#licences) · [Documentation](#documentation)

## Install

Requirements: Python 3.10–3.12 on Windows, macOS or Linux, and about 3 GB of disk for
the recommended extras (PyTorch is the big one). A GPU is optional.

**With [uv](https://docs.astral.sh/uv/) (recommended, reproducible from `uv.lock`):**

```bash
git clone <this repository> manga-arabic && cd manga-arabic
uv venv --python 3.11 .venv
uv sync --locked --extra rapid --extra ocr --extra gui
```

**With pip:**

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[rapid,ocr,gui]"
# pip cannot drop the second OpenCV that some OCR packages pull in:
pip uninstall -y opencv-python opencv-contrib-python
pip install --force-reinstall opencv-python-headless
```

Optional extras:

| Extra | Adds |
|---|---|
| `rapid` | Chinese OCR |
| `ocr` | Japanese/Korean OCR (EasyOCR) |
| `lama` | high-quality inpainting |
| `gui` | the web interface |
| `offline-mt` | local translation models |
| `manga` | manga-ocr for Japanese |
| `paddle` | PaddleOCR |
| `gpu` | CUDA for ONNX models |

The core install without extras still runs the classical detector, OpenCV inpainting,
online translation and the Arabic typesetting.

Then check the installation:

```bash
manga-arabic doctor     # versions, OpenCV sanity, RAQM, fonts, models, network
manga-arabic demo       # generates a page, runs the whole pipeline, prints the output path
```

`demo` needs no network. It uses a bundled translation memory, so a working demo proves
the full pipeline: detection, OCR, inpainting, translation and Arabic lettering.

## Quick start

```bash
# a folder of pages → out/ (PNG), source language auto-detected
manga-arabic translate chapter01/ -o out/

# a CBZ back into a CBZ, Japanese source, your glossary of names
manga-arabic translate volume1.cbz -o out/ --source ja --format cbz --glossary names.yaml

# the local GUI (opens http://127.0.0.1:7860)
manga-arabic gui
```

Each page produces:

| File | What it is |
|---|---|
| `out/<same relative path>/<page>_ar.png` | the translated page |
| `<page>_ar.mangaar.json` | the page's sidecar (regions, OCR, translations, layout, flags) |
| `.mangaar/` | work images, so the page can be re-rendered without re-running the heavy stages |
| `out/report.md` and `out/report.json` | per-page status, warnings and timings |

To fix a translation, edit it in the GUI (**Review & Edit**), or change the sidecar and
run `manga-arabic rerender out/page01_ar.mangaar.json`.

## Command line

| Command | Purpose |
|---|---|
| `translate INPUT… -o OUT` | translate images, folders or CBZ/ZIP archives |
| `rerender PAGE.mangaar.json… [--font F]` | re-typeset pages from their sidecars (after edits) |
| `gui [--host 127.0.0.1] [--port N]` | start the local web GUI |
| `demo [-o DIR]` | installation smoke test on a generated page |
| `doctor` | environment self-check |
| `models {list,download,verify} [names]` | manage model downloads (`download local-mt` for offline translation) |
| `fonts {list,check}` | list fonts and verify their Arabic glyph coverage |

`translate` flags:

| Flag | Values / meaning |
|---|---|
| `--source` | `auto`, `ja`, `ko`, `zh` |
| `--preset` | `fast`, `balanced`, `quality` |
| `--reading-order` | `auto`, `manga_rtl`, `comic_ltr`, `webtoon_ttb` |
| `--providers` | comma list, e.g. `tm,google,mymemory,local` |
| `--glossary FILE` | protected names and terms (JSON/YAML/CSV) |
| `--tm FILE` | translation memory (JSON/YAML/CSV) |
| `--font NAME` | Arabic font key (see `fonts list`) |
| `--digits` | `western`, `arabic_indic` |
| `--sfx` | `skip`, `translate` |
| `--erase-untranslated` | erase text that could not be translated instead of keeping the original |
| `--format` | `png`, `jpg`, `webp`, `cbz` |
| `--offline` | never use the network |
| `--resume` | skip pages already done with the same settings |
| `--force` | reprocess everything |
| `--debug` | write per-page debug images |
| `--device` | `auto`, `cpu`, `cuda`, `mps` |
| `--config FILE` | extra YAML configuration (repeatable) |

**Exit codes:** `0` every page succeeded · `2` partial success (some pages were skipped
or failed; see `report.md`) · `1` nothing could be processed.

Pages are processed in natural order (page2 before page10). Every default lives in the
commented [`configs/default.yaml`](configs/default.yaml). Settings are layered:
defaults → preset → your `--config` YAML → environment (`MANGAAR_CACHE_DIR`,
`MANGAAR_OFFLINE`, `MANGAAR_DEVICE`, `MANGAAR__section__key`) → flags.

## GUI

`manga-arabic gui` starts a local page on `127.0.0.1` (never shared publicly). The
launchers `packaging/run_gui.sh` and `packaging/run_gui.bat` do the same.

- **Translate:** upload pages, a folder or CBZ/ZIP files and press **Translate**. A
  progress bar and a **Cancel** button are available while it runs. When it finishes,
  compare before/after and download a ZIP.
- **Review & Edit:** a table of every region with its source text and editable Arabic
  text, font, size and *skip* columns. **Re-render page** applies your edits in seconds.
  **Re-translate region** asks the providers again.
- **Settings:** preset, device, providers, font, digits, SFX, output format, glossary
  and translation memory.
- **Diagnostics:** the `doctor` report and the recent log.

## Presets

| Preset | Detection | Inpainting | OCR | Use it for |
|---|---|---|---|---|
| `fast` | classical | solid fill + OpenCV Telea/NS | first good engine | quick drafts, slow machines |
| `balanced` (default) | classical (`--detector hybrid` adds a neural detector for text over artwork) | solid fill where uniform, LaMa for textures when installed, Telea otherwise | first good engine | everyday use |
| `quality` | as balanced | LaMa wherever needed, plus a residual-text check | cross-checks two OCR engines | final output |

On a 4-core CPU a typical page takes about 0.5 s (fast) to a few seconds (quality with
LaMa); the details are in [`docs/QUALITY_REPORT.md`](docs/QUALITY_REPORT.md).

## Translation

The default provider chain is `tm → google → mymemory → libretranslate → local`:
- `tm`: your translation memory;
- `google`: the Google web endpoint via deep-translator (no key);
- `mymemory`: the MyMemory free tier;
- `libretranslate`: only when you set `translate.libretranslate_url`;
- `local`: offline models.

Every network call has:
- a hard deadline;
- a client-side rate limiter;
- exponential backoff with jitter that honours `Retry-After`;
- a per-provider circuit breaker.

Results are cached in SQLite, so re-runs cost nothing. A page is sent as one request
when possible, which gives the translator context; if a reply is malformed, MangaAR
falls back to per-bubble requests. Outputs that are empty, unchanged, in the wrong
script or missing glossary terms are rejected, and the next provider is tried.

A region that no provider can translate is flagged `UNTRANSLATED` and keeps its original
pixels (unless you pass `--erase-untranslated`). You can then fill it in the GUI.

### Offline use
Install the local models once while online:

```bash
pip install -e ".[offline-mt]"             # or: uv sync --extra offline-mt
manga-arabic models download local-mt      # Marian OPUS-MT ja/ko/zh→en + en→ar (CC-BY-4.0)
manga-arabic translate pages/ -o out/ --offline
```

A translation memory (`--tm`) also works fully offline. Without either, `--offline`
still runs everything else: regions stay untranslated and the report explains what to
install.

## Models and downloads

Models download on first use into the cache directory, with resume and SHA-256
verification. The cache is `MANGAAR_CACHE_DIR`, or the platform user cache by default.
Nothing large is bundled.

| Model | Used for | Licence |
|---|---|---|
| EasyOCR CRAFT + recognisers (GitHub releases) | ja/ko/zh OCR, optional detector | Apache-2.0 |
| PP-OCR ONNX (bundled in `rapidocr_onnxruntime`) | Chinese OCR, text detector | Apache-2.0 |
| LaMa big-lama TorchScript | textured-background inpainting | Apache-2.0 |
| manga-ocr (Hugging Face) | Japanese OCR | Apache-2.0 |
| OPUS-MT Marian (Hugging Face) | offline translation | **CC-BY-4.0** (attribution) |
| M2M100 418M (Hugging Face, optional) | offline direct translation | MIT |
| comic-text-detector weights | optional detector (`--detector ctd`) | **GPL-3.0**, opt-in only |

To download ahead of time, run `manga-arabic models download default` (EasyOCR and
LaMa). Add `local-mt` for offline translation and `fonts` for the CJK fixture fonts
used by `demo` and the tests.

## Troubleshooting

Start with `manga-arabic doctor`. Common fixes:
- **"two OpenCV distributions installed":** run the pip commands in [Install](#install).
- **Arabic looks disconnected or backwards:** this should be impossible with the bundled
  fonts, which the tests verify. If you pass `--font` with an unusual font, run
  `manga-arabic fonts check`.
- **Everything is `UNTRANSLATED`:**
  - online providers are rate-limited or blocked on your network: retry later with
    `--resume`, or set up offline translation;
  - with `--offline`: install `local-mt`.
- **A page was skipped:** the file is corrupt or not an image; `report.md` says why.
- **Out of memory on huge strips:** MangaAR retries with smaller tiles automatically. Try
  `--device cpu` or the `fast` preset.

Every edge case, with its symptom, automatic behaviour and fix, is in
[`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md).

## Responsible use

- **Copyright.** Translating a comic creates a derivative work. Use MangaAR on content
  you own, content whose licence allows translation, or content you have permission to
  translate. Do not use it to redistribute pirated scans. Support the creators and the
  official releases.
- **Translation quality.** Machine translation makes mistakes, especially with names,
  jokes, honorifics and slang. Review important pages in the GUI before sharing them,
  and use a glossary for names.
- **Online services.** Free translation endpoints are shared resources. MangaAR limits
  its request rate and backs off when asked; do not modify it to circumvent the limits
  or access controls of any service. For heavy use, run offline or host your own
  LibreTranslate.
- **Privacy.** In online mode, OCR text (never images) is sent to the translation
  providers you enable. Use `--offline` for private material.
- **No telemetry.** MangaAR collects nothing. Gradio analytics are disabled, and the GUI
  binds to 127.0.0.1 only.

## Licences

- **MangaAR's own code:** MIT ([LICENSE](LICENSE)).
- **Bundled fonts:** SIL Open Font License 1.1. Licence texts are in
  `src/manga_ar/assets/fonts/licenses/`.
- **Dependencies with obligations:**
  - `python-bidi` (LGPL-3.0): used as an unmodified library.
  - OPUS-MT models (CC-BY-4.0): attribution is required when you publish their output.
  - comic-text-detector weights (GPL-3.0): opt-in only, never downloaded by default.
- **NVIDIA CUDA runtime libraries** (proprietary but redistributable): on Linux the
  PyPI `torch` wheel pulls them in. Only the torch-based extras are affected (`ocr`,
  `lama`, `manga`, `offline-mt`). For a fully open-source install, first run
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.
- NLLB (CC-BY-NC) is **not** used.

The full list is in [`docs/THIRD_PARTY_LICENSES.md`](docs/THIRD_PARTY_LICENSES.md).

## Documentation

| Document | Contents |
|---|---|
| [USER_GUIDE](docs/USER_GUIDE.md) | workflows, glossary/TM formats, editing, webtoons, configuration |
| [ARCHITECTURE](docs/ARCHITECTURE.md) | stages, data model, sidecar, typesetting engine |
| [TROUBLESHOOTING](docs/TROUBLESHOOTING.md) | every edge case (E1–E21) and install problems |
| [DEPENDENCIES](docs/DEPENDENCIES.md) | pinned packages and why each exists |
| [THIRD_PARTY_LICENSES](docs/THIRD_PARTY_LICENSES.md) | licence report (packages, models, fonts) |
| [QUALITY_REPORT](docs/QUALITY_REPORT.md) | benchmark: detection, OCR, inpainting, typesetting, failover, timings |
| [DECISIONS](docs/DECISIONS.md) | design decisions and waivers |
| [ENV_AUDIT](docs/ENV_AUDIT.md), [PLAN](docs/PLAN.md), [STATUS](docs/STATUS.md), [CHANGELOG](CHANGELOG.md) | project records |

### Development

```bash
uv sync --locked --extra dev --extra rapid --extra ocr --extra gui
.venv/bin/python -m pytest -q                    # unit + fast E2E (no network, < 60 s)
.venv/bin/python -m pytest -q -m integration     # real local models (OCR, LaMa, ARVS L7)
.venv/bin/python -m pytest -q -m slow            # CLI E2E with real stages, 50-page soak
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy
.venv/bin/python scripts/benchmark.py            # regenerates docs/QUALITY_REPORT.md
.venv/bin/python scripts/make_specimen_sheet.py  # Arabic font specimen sheets (visual check)
```
