# MangaAR — Build Plan

## Reading of the requirements
Build a FOSS, offline-capable CLI + local Gradio GUI (`manga-arabic`) that translates
manga/manhwa/manhua/comic pages to Arabic via
Load → Detect/segment → OCR → Inpaint → Translate → Typeset (RTL, shaped, fitted) → Export.
Every stage sits behind an interface with a production impl and a test fake; a per-page
sidecar JSON makes re-translation/re-typesetting possible without re-running vision stages.
Batch jobs never crash on one bad page (exit 0/2/1 + report). Arabic typesetting is the
highest-risk component and gets its own verification suite (ARVS L1–L8).

## Environment facts that shape the plan (see ENV_AUDIT.md)
- Linux x86_64, 4 CPU, 15 GB RAM, no GPU, ~30 GB free disk, Python 3.10/3.11/3.12, uv 0.8.
- Egress: PyPI + GitHub (raw + release assets) reachable. **Blocked**: huggingface.co,
  translate.google.com, api.mymemory.translated.net, fonts.google.com, Paddle BOS, ModelScope,
  argos-net, OPUS-MT object store, fbaipublicfiles, download.pytorch.org.
- Consequence: manga-ocr, HF detector, Marian/M2M100/NLLB weights and all live translation
  endpoints cannot be exercised here. They are implemented behind interfaces, tested with
  fakes, and their live gates are waived with reasons in DECISIONS.md.

## Ranked risks → mitigation
1. Arabic rendering wrong (reversed lines, broken joining, tofu) → A1–A14 invariants, ARVS
   with negative controls, RAQM cross-oracle (RAQM is available: libraqm 0.10.5).
2. No reachable translation/local-MT here → full resilience suite with fakes; translation
   memory (TM) provider so the demo is honest and offline; waiver + instructions.
3. OCR on vertical JA without manga-ocr → EasyOCR `ja` + vertical→horizontal reflow;
   RapidOCR (bundled PP-OCR ONNX, no download) for ZH; EasyOCR for KO.
4. Dependency conflicts (duplicate OpenCV, torch/CUDA wheels, OpenCV 5, Gradio 6,
   transformers 5) → verify APIs in spikes; uv overrides; `doctor` detects duplicates.
5. Detection quality → mandatory classical baseline + PP-OCR DB detector (Apache-2.0,
   bundled) + optional comic-text-detector ONNX (GPL-3.0, opt-in only).
6. Scope → strict phase gates, small commits, STATUS.md kept current.

## Phase schedule
- P0 Preflight, scaffold, spikes SP-A..SP-G (SP-A must pass).
- P1 Foundation: config, logging, errors, schemas, io, models manager, doctor, fixtures.
- P2 Detection + bubble segmentation + OCR (router, langid, reflow, suspicion).
- P3 Inpainting (mask, solid fill, Telea/NS, LaMa, strategy, residual, invariants).
- P4 Translation (providers, resilience, cache, batching, glossary, TM, normalize_ar).
- P5 Arabic typesetting + ARVS + specimen sheet.
- P6 Pipeline, sidecar, report, CLI, Gradio GUI, demo, E2E.
- P7 Hardening, docs, launchers, licences, clean-install test, v0.1.0.

## Assumptions
- Project root = repo root (it was empty). Python 3.11 in `.venv` (uv).
- Arabic fonts (OFL, ~4.5 MB) are vendored; CJK fixture fonts are downloaded by
  `scripts/download_assets.py` (Noto CJK subsets, OFL) with system-font fallback.
- Torch comes from PyPI (CUDA build; CPU works). Models are cached under
  `MANGAAR_CACHE_DIR` (dev/tests: `<project>/.cache/`), never committed.
- Default translation chain: tm → google → mymemory → libretranslate (if URL) → local.
