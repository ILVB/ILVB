# CLAUDE.md — MangaAR working memory

MangaAR = FOSS CLI (`manga-arabic`) + local Gradio GUI translating manga/manhwa/manhua
pages to Arabic: Load → Detect/segment → OCR → Inpaint → Translate → Typeset → Export.
Full spec was given in the kickoff prompt; condensed rules below. Docs live in `docs/`.

## Current phase
See `docs/STATUS.md` (checklist + exact next step). Decisions/waivers: `docs/DECISIONS.md`.

## Commands (run from repo root; never global pip)
- Env: `uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e ".[dev,rapid,ocr,gui]"`
- Always export `UV_CACHE_DIR=$PWD/.cache/uv` and `MANGAAR_CACHE_DIR=$PWD/.cache` in dev.
- Tests (unit, default, <60 s): `.venv/bin/python -m pytest -q`
- Integration (real local models): `.venv/bin/python -m pytest -q -m integration`
- Lint/format: `.venv/bin/ruff check . && .venv/bin/ruff format --check .`
- Types: `.venv/bin/mypy`
- CLI: `.venv/bin/manga-arabic doctor | demo | translate IN -o OUT`
- Fixture fonts: `.venv/bin/python scripts/download_assets.py`
- Specimen: `.venv/bin/python scripts/make_specimen_sheet.py` (open + inspect PNG)

## Hard constraints (priority: constraints > Arabic spec > gates > rest)
- FOSS only, no API keys/accounts/telemetry. Flag GPL/AGPL/NC components
  (comic-text-detector weights GPL-3.0 → opt-in only; NLLB CC-BY-NC → not default).
- Never commit comic pages, weights or big binaries. Fixtures are synthetic (code-generated).
- Models → cache dir (platformdirs; `MANGAAR_CACHE_DIR` override; dev = `.cache/`), SHA-256
  verified; `--offline` honoured.
- Canonical image = RGB uint8 HxWx3 numpy. Convert at library boundaries (BGR/PIL).
- Never pass paths to cv2.imread/imwrite (use np.fromfile+imdecode / imencode+tofile).
- All text I/O explicit UTF-8; stdout/stderr reconfigured with errors="replace".
- Exactly one OpenCV distribution: opencv-python-headless (uv override drops others).
- Types everywhere; ruff + mypy strict on core. No print in library code; use logging.
- Exceptions rooted at MangaArError (ImageLoadError, DetectionError, OcrError,
  InpaintError, TranslationError, TypesetError, ModelUnavailableError, ConfigError).
- No bare except, no swallowed errors, no stubs/TODO/NotImplementedError on delivered paths,
  no mocks in prod code, never weaken tests. Deterministic (seeds).
- Pillow ≥10: getlength/getbbox/textbbox only.

## Arabic typesetting invariants (typeset/)
- A1: normalise → tokenise → wrap LOGICAL string (measure reshaped+bidi line) → per line
  reshape → get_display(base_dir='R') → draw LTR with BASIC. Never reshape whole paragraph.
- A2: Path 2 (RAQM, raw text, direction=rtl) is exclusive; NEVER combine with reshaper/bidi.
- A3: python-bidi 0.6 Rust get_display does NOT mirror brackets → shim applies L2+L4 from
  resolved levels. Force base_dir='R'. Strip harakat.
- A4: per-font reshaper config (unshaped-instead-of-isolated when a font lacks isolated
  presentation forms); runtime cmap check of emitted codepoints; never render tofu.
- A5: symbol fallback (NotoSansSymbols2: ♡♥☆★, NotoSansSymbols: ♪♫).
- A7/A8/A9: binary-search fit, balanced wrap, shape-aware spans; overflow ladder; never clip.
- Morphology on crops: use `detect.geometry.erode` (constant-0 border), never bare
  cv2.erode on tight masks (D-033).
- Page composition: clean image has ALL text inpainted; untranslated/failed regions get
  original pixels back at compose time unless erase_untranslated (typeset/page.py, D-034).

## Pipeline / GUI invariants (pipeline.py, ui/)
- Stages are injected via `Stages` (tests use tests/e2e/fakes.py); build_stages is lazy.
- Sidecar `<stem>_ar.mangaar.json` + `.mangaar/<stem>.{source,clean}.png` + `settings`
  snapshot ⇒ rerender needs no ML (D-036). Exit codes 0/2/1 (D-037). Resume D-038.
- GUI: no logic in callbacks (ui/handlers.py GuiController); queue concurrency 1;
  analytics off before `import gradio`; bind 127.0.0.1.
- Slow real-stage tests: `.venv/bin/python -m pytest -m slow` (RapidOCR + TM, offline).

## Release hygiene
- `.gitignore` patterns for outputs/models are root-anchored (`/models/`); the test
  `tests/unit/test_repo_hygiene.py` fails if a source file is ignored or a weight is tracked.
- Regenerate docs/THIRD_PARTY_LICENSES.md with `scripts/license_report.py`, and
  docs/QUALITY_REPORT.md with `scripts/benchmark.py`.
- Clean-install check: `scripts/clean_install_test.sh` (fresh clone → doctor → demo).
- Deep fuzz: `MANGAAR_FUZZ_EXAMPLES=3000 pytest tests/unit/test_fuzz.py`.

## Environment facts (docs/ENV_AUDIT.md)
- Blocked egress: huggingface.co, translate.google.com, mymemory, Paddle hosts, fonts.google.
- Reachable: PyPI, GitHub releases + raw. Usable models: EasyOCR, RapidOCR (bundled),
  big-lama TorchScript, comic-text-detector ONNX.
- deep-translator has no timeout → executor deadline. Google <5000 chars, MyMemory <500.

## Conventions
- src layout `src/manga_ar/`; one responsibility per module; Protocol/ABC per stage.
- Conventional Commits; tag `phase-N-complete` when a gate passes; branch
  `claude/peaceful-meitner-4wcz28`; push with `git push -u origin <branch>`.
- Update docs/STATUS.md after every milestone.
