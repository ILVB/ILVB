# Architecture

MangaAR is a chain of stages that exchange typed data objects. The CLI and the GUI are
thin front-ends over one orchestration API (`manga_ar.pipeline`). Design decisions are
numbered `D-xxx` in [DECISIONS.md](DECISIONS.md).

```
            ┌────────────── Pipeline.run(inputs, out) ───────────────┐
 inputs ──► │ ImageSource (io/sources.py) → PageJob per page          │
            │   │ load_image_bytes (io/loader.py)       ImageLoadError→ skipped
            │   ▼                                                      │
            │ TextDetector ──► BubbleSegmenter ──► reading order      │ per page:
            │ (detect/*)       (detect/bubble.py)   (reading_order)   │ any other
            │   ▼                                                      │ exception →
            │ language vote → OcrStage (ocr/router.py) per region     │ page "failed",
            │   ▼                                                      │ batch goes on
            │ InpaintStage (inpaint/strategy.py) per region           │
            │   ▼                                                      │
            │ TranslateStage (translate/service.py) per page           │
            │   ▼                                                      │
            │ Typesetter + typeset_page (typeset/*) → composite        │
            │   ▼                                                      │
            │ Export: page image, sidecar JSON, work images, debug     │
            └──────────────────────────────────────────────────────────┘
                                   │
                        report.json / report.md, exit code
```

## Package map (`src/manga_ar/`)

| Module | Responsibility |
|---|---|
| `cli.py`, `__main__.py` | argparse front-end: translate, rerender, demo, gui, doctor, models, fonts |
| `pipeline.py` | `Pipeline` (batch, per-page core, resume, export, CBZ), `Stages`, `build_stages`, `rerender` |
| `report.py` | `PageOutcome`, `BatchReport` (JSON/Markdown, exit codes) |
| `config.py` | layered, strictly validated frozen dataclasses; presets; config hash |
| `schemas.py` | `BBox`, `CropMask` (RLE), `Region`, `PageDocument` (sidecar), flags |
| `errors.py` | exception hierarchy rooted at `MangaArError` |
| `cancel.py` | `CancelToken` (cooperative cancellation) |
| `io/` | `sources` (inputs), `loader` (robust decoding), `archive` (safe CBZ/ZIP), `writer` (atomic writes, encoding), `tiling` (webtoon strips), `naming` |
| `models/` | `registry` (model cards: URL, SHA-256, licence), `manager` (download/resume/verify/lazy load, offline), `device` (auto device, OOM → CPU fallback) |
| `detect/` | `classical` (default, no download), `bubble` (flood-fill segmentation, leak fallback), `reading_order` (XY-cut panels), `tiled` (tiles + seam merge + OOM retry), ML adapters (`rapid_detector`, `craft_detector`, `comic_text_detector`, `hybrid`), `factory` |
| `ocr/` | engines (`easyocr_engine`, `rapid_engine`, `manga_ocr_engine`, `paddle_engine`), `router` (routing, retries, candidate choice), `reflow` (vertical → horizontal), `postprocess`, `langid`, `suspicion` |
| `inpaint/` | `mask` (glyph + halo mask, allowed zone), `strategy` (background classification, method chain), `solid_fill`, `opencv_inpaint`, `lama`, `residual` |
| `translate/` | `providers` (Google/MyMemory/LibreTranslate), `local_mt`, `tm`, `resilience` (deadline, limiter, backoff, breaker), `cache` (SQLite), `batching`, `glossary`, `validate`, `normalize_ar`, `service` (chain orchestration) |
| `typeset/` | `fonts` (registry + coverage), `arabic_text` (reshaper config, bidi shim), `textline` (per-run font fallback, BASIC/RAQM engines), `wrap` (balanced DP wrap), `layout` (geometry, fit search, overflow ladder, contrast), `render`, `page` (page composition), `contrast` |
| `ui/` | `handlers` (all GUI logic, testable), `gradio_app` (layout + wiring only) |
| `synth.py`, `demo.py` | seeded synthetic page generator (tests, benchmark, demo) and the demo command |
| `doctor.py`, `debug.py`, `metrics.py`, `resources.py`, `logging_setup.py` | self-check, debug overlays, evaluation metrics, packaged data, logging |

The repository layout follows the suggested one, with the adaptations listed in D-006:
- fonts are packaged under `src/manga_ar/assets/fonts/`;
- the Hugging Face detector is replaced by ML adapters that can actually be downloaded
  here.

## Stage interfaces

Each stage is a `typing.Protocol` with a production implementation and an in-test fake
(`tests/e2e/fakes.py`):

| Interface | Production | Fake |
|---|---|---|
| `ImageSource` | `io.sources.collect_inputs` | temp files in tests |
| `TextDetector.detect(rgb) → [TextBlock]` | `ClassicalDetector` (+ adapters) | `FakeDetector` |
| `BubbleSegmenter.segment(rgb, blocks, page_id) → [Region]` | `FloodBubbleSegmenter` | `FakeSegmenter` |
| `OcrEngine` / `OcrStage.recognize(page, region, lang)` | EasyOCR/RapidOCR/manga-ocr/Paddle + `OcrRouter` | `FakeOcr` |
| `Inpainter.inpaint(image, mask)` / `InpaintStage.inpaint_region` | solid/Telea/NS/LaMa + `RegionInpainter` | `FakeInpainter` |
| `Translator.translate(text, src, tgt)` / `TranslateStage` | providers + `TranslationService` | fake providers, `FakeTranslator` |
| `Typesetter` | `typeset.layout.Typesetter` (pure) | used directly (deterministic) |
| `Exporter` | `Pipeline._export`, `writer`, `archive.write_cbz` | temp dirs |

`Stages` bundles one of each. `build_stages(cfg, manager)` builds the production set
lazily, so models load on first use. Tests inject fakes: `Pipeline(cfg, stages)`.

## Data model and sidecar

A `Region` carries:
- `id`, `type` (bubble / narration / free_text / sfx), `bbox`, `lines`, `polygon`;
- the masks `bubble_mask`, `text_mask` and `inpaint_mask`, each a `CropMask`: bbox +
  row-major RLE in JSON, never raw arrays;
- `safe_box`, `reading_order`, `source_lang`, `vertical`, `fill_color`;
- `ocr` (engine, text, confidence, alternatives, decorations);
- `translation` (provider, text, attempts), `layout` (font, size, lines, strategy, box,
  colours, ladder);
- `flags`;
- the user's `override` (text, font, size, source language, skip).

`PageDocument` (`<stem>_ar.mangaar.json`, `schema_version: 1`) adds:
- the page size, source SHA-256, language and reading-order mode;
- the config hash and a `settings` snapshot;
- status, warnings and stage timings;
- relative paths to the work images (`.mangaar/<stem>.source.png`, `.clean.png`) and to
  the output (plus the CBZ member).

These fields make a page re-renderable without earlier stages, which is what `rerender`
and the GUI editor rely on (D-036).

## Configuration

`data/default.yaml` holds every key; `configs/default.yaml` is a byte-identical copy for
readers, enforced by a test. The layers are:
1. defaults;
2. preset overlay (`PRESETS` in `config.py`);
3. user YAML files;
4. environment (`MANGAAR_*`);
5. CLI overrides.

Each layer is validated strictly (unknown keys and bad values raise `ConfigError`).
`config_hash()` covers every output-affecting key plus the version and drives
`--resume`.

## Models

`models/registry.py` describes each model:
- URL(s), SHA-256, size;
- licence, with a copyleft/NC flag;
- kind: file, Hugging Face repo, or EasyOCR bundle.

`ModelManager.ensure()` downloads into the cache with resume, retries with backoff,
verifies the hash, checks free disk space and honours offline mode. On failure it raises
`ModelUnavailableError` with manual-download instructions. Stages fall back gracefully:
- ML detectors fall back to the classical one;
- LaMa falls back to OpenCV inpainting;
- OCR engines fall back to the next engine.

## Translation resilience

`translate/resilience.py` wraps every network call:
- **hard deadline** (a daemon thread, because deep-translator exposes no timeout);
- **token-bucket limiter** (1 request/s, burst 2);
- **backoff** with full jitter that honours `Retry-After` (base 1 s, ×2, cap 60 s, up to
  6 attempts);
- **circuit breaker** per provider (opens after 5 failures, 120 s cool-down, half-open
  probe).

`TranslationService`:
1. de-duplicates identical texts and protects glossary terms with `ZQX<n>X` tokens;
2. batches a page into one `[1] … [2] …` request, falling back to per-region requests
   when the markers do not survive;
3. validates every output (script ratio, length, placeholders);
4. walks the provider chain;
5. caches results in SQLite.

All of it is tested with a fake clock (D-021..D-027).

## Arabic typesetting engine

The engine follows the spec's invariants A1–A14:
1. `normalize_ar`, digit policy, harakat stripping, then tokenising into words.
2. Geometry: the padded bubble mask (strategy `shape`), or its largest inscribed
   rectangle (`rect`, used for LEAK_FALLBACK).
3. Fit: integer binary search over font size, then a short upward probe. For each size,
   the text is wrapped on the logical string using the balanced DP in `wrap.py`, with
   per-line spans read from the mask. Candidate lines are estimated from cached word
   widths and verified exactly.
4. Each line is shaped: `reshape` with a per-font explicit configuration, then the bidi
   shim resolves Unicode levels and applies L2 reordering and L4 mirroring. Lines are
   drawn with Pillow's BASIC layout, with a per-run font fallback for symbols. RAQM
   (HarfBuzz) is the alternative path and the test oracle; the two are never combined.
5. Overflow ladder, then WCAG contrast colours and outlines. Everything is drawn on an
   RGBA layer, alpha-composited, and kept inside the image.
6. `typeset/page.py` composes the page. It restores original pixels for regions without
   Arabic text (unless `erase_untranslated` is set) and isolates per-region failures.

The Arabic Rendering Verification Suite (ARVS) checks this engine at eight levels:

| Level | Checks | Where |
|---|---|---|
| L1 | normalisation | `test_normalize_ar` |
| L2 | shaping oracle | `test_arvs` |
| L3 | bidi | `test_arvs` |
| L4 | glyph coverage | `test_arvs` |
| L5 | raster connectivity/direction, with negative controls | `test_arvs` |
| L6 | RAQM cross-engine oracle | `test_arvs` |
| L7 | OCR round-trip | `tests/integration/test_arvs_ocr.py` |
| L8 | visual specimen sheets | `scripts/make_specimen_sheet.py` |

## Failure isolation, concurrency and cancellation

- **Region level:**
  - OCR crash → OCR_FAILED;
  - inpainting failure → SKIPPED (original pixels restored);
  - translation failure → UNTRANSLATED;
  - typesetting failure → TYPESET_FAILED.
- **Page level:** unreadable → `skipped`; unexpected exception → `failed`. The batch
  continues and the report records everything (D-037, D-040).
- **Threads:** processing is sequential by design. Several ML runtimes are not
  thread-safe, and the GUI runs all work on one queue worker.
- **Cancellation:** a `CancelToken` checked between pages and regions.
- **Atomic output:** every file is written to a temp file and renamed, so a crash never
  leaves a truncated output.

## Testing architecture

| Marker | Contents | Command |
|---|---|---|
| (default / `unit`) | unit tests, ARVS L1–L6, fake-stage E2E, GUI handlers + gradio_client smoke, in-process CLI, fuzzing | `pytest -q` (< 60 s, no network, no downloads) |
| `integration` | real OCR engines (CER gates), LaMa, ARVS L7 | `pytest -m integration` |
| `slow` | CLI subprocess E2E with real stages, 50-page soak | `pytest -m slow` |
| `network` | live translation endpoints | `pytest -m network` |

Synthetic fixtures (`manga_ar.synth`) draw pages with ground truth: boxes, masks, text,
language and reading order. No copyrighted page is ever used.
