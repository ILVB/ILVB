# Audit of MangaAR v0.1.0 (OP-1)

| Item | Value |
|---|---|
| Audited commit | `7ac67181e3aa44cc85ba8efea75ee43a998e15f5` (tag `v0.1.0`, local; the remote refuses tag pushes) |
| Audit date | 2026-09-30 |
| Environment | Linux x86_64, 4 CPUs, no GPU, Python 3.11.15 in `.venv` (uv), see `docs/ENV_AUDIT.md` |
| Baseline tag | `v0.1.0` already exists on the audited commit, so no `v0.1.0-baseline` tag was created (OP-1) |

Differences between this repository and the master prompt's assumptions are recorded
in [ADR-0001](adr/0001-v0.1.0-differences-from-master-prompt.md).

## 1. Module map (`src/manga_ar/`, 11,336 lines of Python)

| Package / module | Responsibility |
|---|---|
| `cli.py`, `__main__.py` | argparse CLI `manga-arabic`: `translate`, `rerender`, `demo`, `gui`, `doctor`, `models`, `fonts` |
| `pipeline.py` | orchestration (`Pipeline`, injectable `Stages`, resume, export, CBZ repack, `rerender`) |
| `report.py` | batch report JSON/Markdown, exit codes 0/2/1 |
| `config.py`, `data/default.yaml` | layered strict config (defaults → preset → YAML → env → CLI), presets fast/balanced/quality, config hash |
| `schemas.py` | `BBox`, `CropMask` (RLE), `Region`, `PageDocument` (sidecar schema v1), `Flag` |
| `io/` | `sources` (inputs), `loader` (robust decoding), `archive` (safe CBZ/ZIP), `writer` (atomic writes), `tiling` (webtoons), `naming` |
| `models/` | `registry` (URLs, SHA-256, licences), `manager` (download/resume/verify/offline), `device` (auto device, OOM → CPU) |
| `detect/` | `classical` (default), `bubble` (flood-fill segmentation + leak fallback), `reading_order` (XY-cut panels), `tiled`, ML adapters `rapid_detector`, `craft_detector`, `comic_text_detector`, `hybrid`, `factory` |
| `ocr/` | engines `easyocr_engine`, `rapid_engine`, `manga_ocr_engine`, `paddle_engine`; `router`, `reflow`, `postprocess`, `langid`, `suspicion` |
| `inpaint/` | `mask`, `strategy` (background classifier + method chain), `solid_fill`, `opencv_inpaint`, `lama`, `residual` |
| `translate/` | `providers` (Google/MyMemory/LibreTranslate via deep-translator), `local_mt` (Marian/M2M100), `tm`, `resilience`, `cache`, `batching`, `glossary`, `validate`, `normalize_ar`, `service` |
| `typeset/` | `fonts`, `arabic_text` (reshaper + bidi shim), `textline`, `wrap`, `layout`, `render`, `page`, `contrast` |
| `ui/` | `handlers` (GUI logic), `gradio_app` (Gradio 6 layout) |
| `synth.py`, `demo.py`, `doctor.py`, `debug.py`, `metrics.py`, `resources.py`, `logging_setup.py`, `cancel.py` | synthetic pages, demo, self-check, debug overlays, metrics, packaged data, logging, cancellation |

## 2. Data flow

```
inputs (files / folders / CBZ-ZIP)
  → io.sources.collect_inputs → PageJob            (natural order, relative output dirs)
  → io.loader.load_image_bytes → RGB uint8          (EXIF, modes, bombs, truncation policy)
  → detect.tiled.detect_page(TextDetector)          (tiles for tall strips, seam merge)
  → detect.bubble.FloodBubbleSegmenter → [Region]   (types bubble/narration/free_text/sfx)
  → language vote (OcrRouter.detect_language), reading order (manga_rtl/comic_ltr/webtoon_ttb)
  → OcrRouter.recognize per region                  (engine chain, reflow, suspicion flags)
  → RegionInpainter.inpaint_region per region       (solid / LaMa / Telea-NS, allowed zone)
  → TranslationService.translate_regions per page   (TM → google → mymemory → libretranslate → local)
  → typeset.page.typeset_page                       (fit, overflow ladder, compose)
  → export: page image, sidecar JSON, work images, debug artefacts, report.json/.md
```

Failures are isolated per region (flags OCR_FAILED, SKIPPED, UNTRANSLATED,
TYPESET_FAILED) and per page (`skipped` / `failed`); see D-037, D-040.

## 3. Current implementations

### Detection and OCR
- **Default detector:** the classical, training-free `ClassicalDetector` (adaptive
  threshold, polarity arbitration, container grouping, furigana split, SFX typing).
  Box-level blocks carry a pixel text mask derived by thresholding, not by a
  segmentation model.
- **Optional detectors:** `hybrid` (classical + PP-OCR DB via RapidOCR), `rapid`,
  `craft` (EasyOCR CRAFT) and `ctd` (comic-text-detector ONNX, GPL-3.0 weights, opt-in).
- **Region types:** `bubble`, `narration`, `free_text`, `sfx`. There are no `dialogue` /
  `thought` / `sign` / `credit` classes and no speaker attribution.
- **Reading order:** panel-aware recursive XY-cuts with modes `manga_rtl`, `comic_ltr` and
  `webtoon_ttb`.
- **OCR routing** (config `ocr.engines`):

  | Language | Engine chain |
  |---|---|
  | ja | manga_ocr → easyocr → rapid → paddle |
  | ko | easyocr → paddle |
  | zh | rapid → easyocr → paddle |

  manga-ocr and PaddleOCR weights cannot be downloaded on this host (W-001, W-002).
- **Measured quality** (`docs/QUALITY_REPORT.md`, synthetic held-out seeds 100–103):
  - detection precision = recall = 1.000;
  - CER: ja horizontal 0.000, ja vertical 0.028, ko 0.068, zh 0.000.

### Translation
- **Provider chain:** `tm → google → mymemory → libretranslate → local`:
  - `google` and `mymemory`: free web endpoints via deep-translator;
  - `libretranslate`: only when a URL is configured;
  - `local`: Marian OPUS-MT pivot through English, or M2M100.
- **Resilience:** deadline, token bucket, full-jitter backoff honouring Retry-After,
  circuit breaker, SQLite cache.
- **Batching and checks:** page-level `[n]` batching with an integrity fallback,
  glossary placeholders, output validation (script ratio, length, placeholders) and
  `normalize_ar`.
- **Not present:** no LLM provider, no NLLB, no Series Bible, no semantic memory, no
  context window beyond the page batch, no speaker/gender metadata.
- **On this host:** Google, MyMemory and the Hugging Face weights for local MT are all
  blocked. Only the TM provider actually translates here (W-003, W-004).

### Inpainting
- The mask is glyphs + stray ink + halo, with adaptive dilation clipped to an allowed
  zone (the bubble interior eroded).
- A robust background classifier picks solid fill (uniform backgrounds), big-lama
  TorchScript on context-padded crops (textures), or OpenCV Telea/NS (fallback or
  gradients).
- Compositing is exact: pixels outside the mask are bit-identical.
- A residual-text check re-runs the detector (quality preset).
- Measured: 0 pixels changed outside masks; 0/140 regions with residual text
  (`docs/QUALITY_REPORT.md`).

### Typesetting
- The default BASIC path is arabic-reshaper plus a python-bidi shim that performs UBA L2
  reordering and L4 mirroring. Pillow RAQM is optional.
- Wrapping is a balanced dynamic programme on the logical string, over shape-aware spans
  taken from the padded bubble mask (or the inscribed rectangle).
- Size is chosen by binary search with an upward probe.
- Overflow ladder: spacing → condensed → padding → extend-uniform → hard floor
  (OVERFLOW_RISK).
- Colours follow WCAG contrast, with outlines for free text.
- 11 OFL Arabic fonts plus 2 symbol fonts are vendored.
- **Target is Arabic only:** there is no English typesetting, no hyphenation and no
  "ask translator for shorter text" step.
- ARVS L1–L8 exists: L7 OCR round-trip gives 246/250 samples ≥ 0.8 similarity.

### GUI
- Gradio 6.29 Blocks (`ui/gradio_app.py`) over `ui/handlers.GuiController`.
- Tabs: Translate, Review & Edit, Settings, Diagnostics.
- Single queue worker; binds to 127.0.0.1; analytics disabled.
- **Not** PyQt/PySide. This matters for the Phase 2 framework rule P2.1: the default
  becomes the local FastAPI web app.

## 4. Languages and input formats

| Aspect | v0.1.0 support |
|---|---|
| Source languages | Japanese (`ja`), Korean (`ko`), Chinese (`zh`), plus `auto` (document-level script vote) |
| Target languages | Arabic (`ar`) only (`translate.target: ar`, `TARGET = "ar"`) |
| English source | **not supported** (no Latin OCR route; Latin-only text is PASS_THROUGH) |
| Single image | PNG, JPEG, WebP, BMP, TIFF, GIF (first frame) |
| Folder | yes, recursive, natural order, dot-folders skipped |
| CBZ / ZIP | yes (in memory, zip-slip and zip-bomb guards) |
| PDF | **not supported** |
| Output | PNG, JPEG, WebP, CBZ; sidecar `<stem>_ar.mangaar.json` + work images |

## 5. Test status (commands run in this session)

The default suite ran twice. The first run was on a cold container (restarted 4
minutes earlier): `296 passed, 18 deselected in 90.92s`. The warm re-run:

```
$ .venv/bin/python -m pytest -q
296 passed, 18 deselected in 30.24s
```

Integration and slow suites (`-m "integration or slow"`):

```
$ .venv/bin/python -m pytest -q -m "integration or slow"
SKIPPED [1] tests/integration/test_local_mt_real.py:30: local MT weights unavailable (W-004) ...
SKIPPED [1] tests/integration/test_ocr_engines.py:83: manga-ocr weights unavailable here (W-001) ...
13 passed, 2 skipped, 299 deselected in 204.24s (0:03:24)
```

The integration skips are the Hugging Face models blocked by the network policy
(W-001 manga-ocr, W-004 local MT). There is no CI configuration, no pre-commit
configuration and no `pip-audit` in v0.1.0.

## 6. Dependency versions (from `uv.lock`)

| Package | Version | Package | Version |
|---|---|---|---|
| numpy | 2.4.6 (py3.11) | opencv-python-headless | 5.0.0.93 |
| pillow | 12.3.0 | fonttools | 4.66.1 |
| arabic-reshaper | 3.0.1 | python-bidi | 0.6.11 |
| deep-translator | 1.11.4 | huggingface-hub | 1.33.0 |
| rapidocr-onnxruntime | 1.4.4 | onnxruntime | 1.30.0 |
| easyocr | 1.7.2 | torch / torchvision | 2.14.0 / 0.29.0 |
| manga-ocr | 0.1.16 | transformers / sentencepiece | 5.17.0 / 0.2.2 |
| paddleocr / paddlepaddle | 2.10.0 / 3.3.1 | gradio / gradio-client | 6.29.0 / 2.7.1 |
| pytest | 9.1.1 | hypothesis | 6.168.3 |
| ruff | 0.16.9 | mypy | 2.3.1 |

The full pinned set (167 packages) is in `uv.lock`; installed licences are in
`docs/THIRD_PARTY_LICENSES.md`.

## 7. Known defects and limitations

1. **Hatching blind spot:** glyphs drawn straight onto dense 1-px hatching without a halo
   can be missed by the classical detector and by the residual check (D-019).
2. **Korean OCR:** relies on EasyOCR (CER 0.068 on synthetic pages).
3. **Symbol size:** ♪ from Noto Sans Symbols renders smaller than the Arabic text.
4. **ICC on rerender:** `rerender` does not re-embed the source ICC profile.
5. **Online translation unverified:** Google/MyMemory are blocked here; the live smoke
   test was never run (W-003).
6. **Local MT unverified:** the Marian/M2M100 weights are blocked here (W-004).
7. **No segmentation-model masks:** text masks come from thresholding and morphology; ctd
   gives masks but is GPL and opt-in.
8. **No PDF input, no English source, no non-Arabic target.**
9. **Coarse region taxonomy:** no dialogue/thought/sign/credit distinction and no
   speaker attribution, so gender agreement cannot be controlled.
10. **Translation context** is limited to one page batch; there is no Series Bible and
    no memory.
11. **NVIDIA libraries:** on Linux, PyPI torch pulls proprietary NVIDIA CUDA libraries
    (D-043).
12. **No PyInstaller spec** (D-045).

## 8. Licence status

| Class | Status |
|---|---|
| MangaAR code | MIT |
| Python packages (128 installed) | `docs/THIRD_PARTY_LICENSES.md` (generated by `scripts/license_report.py`). Flagged: python-bidi LGPL-3.0 (unmodified); MPL-2.0 file-level (certifi, tqdm, orjson, hypothesis, pathspec); **17 proprietary NVIDIA CUDA wheels + cuda-toolkit** via torch on Linux (D-043) |
| Models | EasyOCR (Apache-2.0), PP-OCR in RapidOCR (Apache-2.0), big-lama (Apache-2.0), manga-ocr (Apache-2.0, not downloadable here), OPUS-MT (CC-BY-4.0, attribution), M2M100 (MIT), comic-text-detector weights (**GPL-3.0, opt-in**) |
| Fonts | 11 Arabic + 2 symbol fonts, all SIL OFL-1.1, licence texts in `src/manga_ar/assets/fonts/licenses/`; CJK fixture fonts (Noto Sans JP/KR/SC subsets, OFL-1.1) downloaded, never committed |
| Datasets | none; all fixtures are code-generated synthetic pages (`manga_ar.synth`) |

## 9. Environment constraints relevant to v0.2.0

Probed on 2026-09-30 from this host (curl, HTTP status):

| Resource | Needed for | Reachable |
|---|---|---|
| huggingface.co | NLLB-200, GGUF LLMs, sentence-transformers, manga-ocr | **no** (000) |
| download.pytorch.org | torchvision AlexNet/VGG backbones for LPIPS | **no** (000) |
| PyPI (ctranslate2, llama-cpp-python sdist, uharfbuzz, lpips, sacrebleu, …) | libraries | yes |
| raw.githubusercontent.com google/fonts | Comic Neue, Bangers, Noto Kufi Arabic, Reem Kufi | yes |
| raw.githubusercontent.com nltk_data | WordNet (METEOR) | yes |
| blambot.com | Wild Words | **no**; licence restricts redistribution anyway |
| abetlen.github.io | prebuilt llama-cpp-python wheels | **no** (build from sdist instead) |

As a result, the mandatory Phase 1 translation models (NLLB-200, GGUF LLMs, embedding
models) and the LPIPS backbone cannot be obtained on this host. This is escalated to
the human (see the consolidated request in `docs/LEDGER.md`).
