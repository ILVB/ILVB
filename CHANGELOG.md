# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-09-30

First release.

### Added
- **Pipeline:** load → detect/segment → OCR → inpaint → translate → typeset → export.
  - Failures are isolated per region and per page.
  - Each page gets a sidecar JSON (`<page>_ar.mangaar.json`, schema v1) plus work
    images, so pages can be re-rendered without the heavy stages.
  - `--resume` uses the config hash and source hash; degraded pages retry translation
    only.
  - CBZ repacking, debug artifacts, cooperative cancellation.
  - Batch reports `report.json` / `report.md` with exit codes 0/2/1.
- **Inputs:** PNG, JPEG, WebP, BMP, TIFF, folders (recursive, natural order) and
  CBZ/ZIP (zip-slip and zip-bomb safe).
  - EXIF rotation; palette/alpha/CMYK/16-bit/animated images; truncated-file policy;
    decompression-bomb guard.
  - Webtoon tiling with seam merging and an out-of-memory retry.
- **Detection:**
  - classical detector (no download) with polarity arbitration, furigana handling and
    SFX typing;
  - flood-fill bubble segmentation with leak protection;
  - panel-aware reading order (`manga_rtl`, `comic_ltr`, `webtoon_ttb`);
  - optional ML detectors: `hybrid`, `rapid`, `craft`, and `ctd` (GPL weights, opt-in).
- **OCR:**
  - engines: EasyOCR (ja/ko/zh), RapidOCR PP-OCR (zh), manga-ocr (ja), PaddleOCR
    adapters;
  - routing with retries, vertical-to-horizontal reflow and suspicion checks;
  - document-level source-language vote.
- **Inpainting:**
  - glyph + halo masks confined to an allowed zone;
  - solid fill for uniform bubbles, LaMa (TorchScript) for textures, OpenCV Telea/NS
    fallback;
  - residual-text check;
  - pixels outside the mask are bit-identical to the input.
- **Translation:**
  - providers: TM → Google (deep-translator) → MyMemory → LibreTranslate → local
    Marian/M2M100;
  - resilience: deadline, token bucket, backoff with full jitter honouring
    `Retry-After`, circuit breaker;
  - SQLite cache, page batching with an integrity fallback, glossary placeholders,
    output validation, `normalize_ar`.
- **Arabic typesetting:**
  - wrapping on logical text, per-line shaping, bidi shim with UBA L2 reordering and L4
    mirroring;
  - per-font reshaper configuration and per-run symbol fallback;
  - balanced shape-aware fit (binary search), overflow ladder, WCAG colours, outlines
    for free text;
  - optional RAQM path.
  - 11 bundled OFL fonts.
- **CLI `manga-arabic`:** `translate`, `rerender`, `gui`, `demo`, `doctor`, `models`,
  `fonts`.
- **Local Gradio GUI:** Translate, Review & Edit, Settings, Diagnostics. One queue
  worker, cancel, temp-workspace clean-up, analytics disabled, bound to 127.0.0.1.
- **Quality tooling:**
  - ARVS L1–L8 (L7 OCR round-trip: 246/250 samples ≥ 0.8 similarity);
  - synthetic page generator with ground truth, `scripts/benchmark.py`,
    `scripts/make_specimen_sheet.py`;
  - fuzz tests, a 50-page soak test, `scripts/license_report.py`;
  - clean-install test scripts, GUI launchers.

### Known limitations
- Glyphs drawn straight onto dense hatching without a halo can be missed by the
  classical detector (use `--detector hybrid`).
- Korean OCR relies on EasyOCR (CER ≈ 6 % on synthetic pages).
- On Linux the PyPI `torch` wheel pulls proprietary (redistributable) NVIDIA CUDA
  libraries. Install the CPU wheel for a fully open-source environment (see README).
- Not verified on the build host, whose network policy blocks these downloads:
  manga-ocr, PaddleOCR models, local MT weights and live online translation (DECISIONS
  W-001…W-004).

[0.1.0]: https://github.com/ilvb/ilvb/releases/tag/v0.1.0
