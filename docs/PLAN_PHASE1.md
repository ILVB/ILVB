# PLAN — Phase 0 (remaining) and Phase 1 → v0.2.0

Order: 0.2–0.6 → 1.0 → 1.1 → 1.3 → 1.2 → 1.5 → 1.4 (gate), as the prompt recommends.
Every step follows OP-3:
1. acceptance criteria;
2. failing tests first;
3. smallest change;
4. fast tests, ruff and mypy;
5. dev-split micro-benchmark;
6. commit (`Phase:`/`Step:` trailers);
7. ledger entry.

Rollback for every step is `git revert <sha>` plus the step's feature flag set to
`legacy`, unless a different rollback is stated. Paths are listed in `docs/PHASE_MAP.md`.
New runtime behaviour is off in profile `legacy`, which reproduces v0.1.0 (OP-10).

## Phase 0

| ID | Goal | Files | Tests | Acceptance |
|---|---|---|---|---|
| 0.2.1 | Baseline worktree `.baseline/` at `v0.1.0` | `tools/baseline_worktree.py` | worktree HEAD == `v0.1.0` SHA; refuses a dirty or moved worktree | `python tools/baseline_worktree.py --verify` exits 0 |
| 0.2.2 | Common adapter: one harness runs baseline and candidate on identical inputs | `benchmarks/adapters/{base,baseline,candidate}.py`, `benchmarks/adapters/runner.py` | both adapters return schema-valid `PageResult` for one synthetic page; the baseline adapter imports code from `.baseline/src` (asserted by `__file__`) | JSON schema validation passes for both |
| 0.3.2 | Synthetic generator v2, self-contained (does not import `manga_ar`, so product changes cannot move the yardstick) | `benchmarks/generators/*.py`, pinned font manifest | determinism (identical SHA-256 across runs); all 10 categories present; ≥ 400 regions; ground truth (text, polygons, masks, bubble interior, clean background) consistent | `python -m benchmarks.generators.build --check` |
| 0.3.3 | Dataset manifest and splits dev/val/test, stratified by category, page-level (no leakage); test hashes recorded; sealed access | `benchmarks/data/synthetic_v1/manifest.json`, `benchmarks/datasets.py` | no page in two splits; stratification tolerance; test loader refuses access without the gate capability | manifest committed; SHA-256 of the test split recorded in `benchmarks/data/synthetic_v1/test.sha256` |
| 0.3.4 | Translation references for synthetic sources: **SILVER** (authored by a model not under test), plus drafts of the nuance/behaviour/injection/refusal suites marked pending human verification | `benchmarks/data/text/*.yaml`, ADR-0002 | schema tests; every silver item labelled; nuance suite ≥ 60 items | reports print `SILVER-REFERENCE, comparative only` |
| 0.3.5 | Real gold-data ingestion (for when the human delivers) | `tools/ingest_gold.py` | templates round-trip; schema errors reported with line numbers; split by series/page | `--check` on the templates exits 0 |
| 0.4 | Environment capture on every run | `benchmarks/env.py` | keys present; model checksums listed | `benchmarks/env.json` written by every harness run |
| 0.2.3 | Harness `run_benchmark.py` + metric library | `benchmarks/run_benchmark.py`, `benchmarks/metrics/*.py`, `[bench]` extra | metric unit tests against hand-computed values (CER, WER, P/R/F1 @IoU0.5, SSIM, PSNR, ink-outside-safe, raggedness, residual text, BLEU/chrF++ via sacrebleu, METEOR adapted); seeded, content-hashed outputs; `--split test` refused without the capability | a dev run for both versions writes raw per-item JSON + images |
| 0.2.4 | Freeze the baseline for dev/val/test (test sealed: written without printing per-item outputs) | `benchmarks/results/baseline_v0.1.0.json` + `.sha256` | the gate refuses a modified file (negative control) | hash recorded in the ledger |
| 0.5.1 | `tools/audit_phase_order.py` | `tools/`, `tests/tools/` | negative controls: a fabricated `Phase: 2` commit, a Phase 2 path, a `spacy` dependency and a missing trailer each make it fail | exits 0 on the real history |
| 0.5.2 | `tools/assert_gate.py` | `tools/`, `tests/tools/` | fails without `.gates/phase1.approved`, on a SHA mismatch, a non-PASS verdict, a non-ancestor commit, or a Phase-1 file changed after the gate | all negative controls fail as designed |
| 0.5.3 | pre-commit + CI | `.pre-commit-config.yaml`, `.github/workflows/ci.yml` | hooks run locally | ruff, mypy, pytest, `pip-audit`, both tools, and the edge-case matrix check |
| 0.5.4 | `docs/PHASE_MAP.md`, `docs/EDGE_CASE_MATRIX.md` + `tools/check_edge_matrix.py` | docs, tools | the checker fails when an E-ID has no test | CI step green |
| 0.5.5 | `docs/LICENSES.md` register (generated + curated models/fonts/datasets) | `scripts/license_report.py` | the generator test lists every registry model/font | G-LIC-1 check function exists |
| 0.6 | `benchmarks/budgets.yaml` from baseline dev/val measurements (immutable afterwards) | `benchmarks/budgets.yaml`, ADR-0003 | schema test; the gate reads it | committed before any Phase 1 implementation commit |

## Phase 1

### 1.0 OCR and detection (feeds G-OCR)

| ID | Goal | Acceptance (dev/val) |
|---|---|---|
| 1.0.1 | Typed `TextRegion` (pydantic): id, page, polygon, mask ref, text, confidence, script/lang, type ∈ {dialogue, thought, narration, sfx, sign, credit}, reading_order, speaker?, bubble polygon?; legacy `Region` ↔ `TextRegion` adapter; profile flag `engine.profile: legacy\|v2` | round-trip tests; legacy profile output byte-identical to v0.1.0 on the dev set |
| 1.0.2 | Baseline measurement per category on dev: CER/WER, detection P/R/F1 @IoU0.5 | per-category table in the ledger |
| 1.0.3 | Pixel-level text masks from a segmentation model: PP-OCR DB probability maps (Apache-2.0, bundled) vs comic-text-detector (GPL-3.0, opt-in) vs the classical detector | mask IoU vs ground truth on dev improves; ADR-0004 (detector/mask selection) |
| 1.0.4 | Non-destructive preprocessing: adaptive upscaling of small text, contrast normalisation, screentone suppression, deskew/rotation, vertical text | CER per category on dev; each transform kept only if it helps on val |
| 1.0.5 | Recognizer selection per script, including an English/Latin route (EasyOCR `en`, PP-OCR) | CER per script on val; ADR-0004 update |
| 1.0.6 | Constrained post-OCR correction of low-confidence tokens (edit distance ≤ 2, every edit logged); kept only if CER improves on val (no Phase-2-only resources) | CER-judged; may be dropped with a ledger justification |
| 1.0.7 | Region typing (dialogue/thought/narration/sfx/sign/credit) + SFX classifier (E-01) | type accuracy on dev/val |
| 1.0.8 | Reading order: configurable RTL/LTR, inferred from metadata (E-16), fixture layouts | unit tests on fixture layouts |

### 1.1 Translation accuracy engine (feeds G-TR)

| ID | Goal | Acceptance |
|---|---|---|
| 1.1.1 | `TranslationProvider` interface + pydantic result (translation, confidence, provenance); `LegacyProvider` wraps v0.1.0 `TranslationService` | contract tests; the legacy profile is unchanged |
| 1.1.2 | Router (primary → fallback → last resort), circuit breaker, per-region provenance; content-addressed cache (provider, model hash, prompt version, context hash, source) | fallback-chain and determinism tests |
| 1.1.3 | Output defences (script ratio, length bounds, refusal/boilerplate, meta-commentary, repetition loops, source leakage) + injection hygiene (delimited data) | refusal and injection suites 100 % on fakes (E-12, E-13) |
| 1.1.4 | `NLLBProvider` (CTranslate2, NLLB codes, batching, beams) | unit tests with a fake CT2 translator; the integration test needs model weights (blocked, ADR-0001 #7) |
| 1.1.5 | `LocalLLMProvider` (llama.cpp server/python; deterministic; JSON-schema output; length-aware) | unit tests with a fake endpoint; the integration test needs a GGUF (blocked) |
| 1.1.6 | Series Bible (SQLite, YAML export/import): characters, locked glossary, places, style, chapter summary; locked terms injected and verified with repair/retry | E-14, E-17 tests; glossary adherence on fakes |
| 1.1.7 | Semantic memory: per-series vector index (SQLite-backed); top-k retrieval + sliding window; versioned, resettable; zero cross-series leakage | isolation test; embedding backend ADR (sentence-transformers blocked → deterministic local fallback) |
| 1.1.8 | Chapter pre-pass: recurring terms/entities (frequency + patterns) → `glossary_proposals.yaml` | unit tests |
| 1.1.9 | Prompt assembly (rules → bible → memory → recent → batch → constraints: MSA, gender/number, honorifics, SFX policy, max length from the typesetter capacity); output contract JSON {id, translation, glossary_used[], confidence}; validate → repair → retry (≤ 2) → fallback | schema-repair tests; E-07, E-15 |
| 1.1.10 | Suites: nuance (≥ 60, pending human verification), Arabic behaviour, injection, refusal; benchmark wiring (BLEU, chrF++, METEOR adaptation, paired bootstrap) | suites run in the harness; the report labels silver refs |

### 1.3 Inpainting (feeds G-INP)

| ID | Goal | Acceptance (dev/val) |
|---|---|---|
| 1.3.1 | Mask v2: stroke-width-adaptive dilation, polarity (E-18), intersection with the bubble/region mask | residual-text rate and SSIM on dev |
| 1.3.2 | Router (flat fill / LaMa / Telea-NS) with thresholds tuned on dev, validated on val; decisions logged | ADR-0005; SSIM ≥ 0.95 on flat categories (dev) |
| 1.3.3 | LaMa ROI policy: context ≥ 2× text height, model-friendly snapping, native resolution, overlapped blended tiles, composite masked + feather ring only | bit-exact outside mask + ring; tile-seam test |
| 1.3.4 | Mode preservation end to end (grayscale, palette, 16-bit, alpha, ICC, EXIF) (E-10) | round-trip tests |
| 1.3.5 | Residual-text loop (≤ 2 passes, needs_review) | residual rate ≤ baseline on dev |
| 1.3.6 | Device fallback, OOM tiling, deterministic seeds (E-11) | fault-injection tests |

### 1.2 Typesetting (feeds G-TYPE)

| ID | Goal | Acceptance |
|---|---|---|
| 1.2.1 | Font registry (`src/manga_ar/assets/fonts/registry.yaml`: family, scripts, licence, URL, SHA-256, roles, variants, metric overrides) + deterministic selector; download/verify script for Comic Neue, Bangers, Noto Kufi Arabic / Reem Kufi; a Wild Words install/verify step (not bundled) | selector unit tests; ADR-0006 font mapping from a specimen sheet |
| 1.2.2 | HarfBuzz shaping (uharfbuzz) as the primary Arabic path, reshaper+bidi as the fallback; cross-check corpus ≥ 50 strings | 100 % glyph-sequence agreement; zero `.notdef` (E-06) |
| 1.2.3 | Safe region from the mask: distance transform, opening (tails excluded), proportional margin; text-on-art = dilated text hull (E-02, E-03) | property test: ink ⊆ safe region |
| 1.2.4 | Ink-based fitting, chord-width DP line breaking, size ceiling from the original text size (E-09), page consistency ±20 % | raggedness vs baseline on dev |
| 1.2.5 | Hyphenation: pyphen for English only, never Arabic; kashida flag default off | hyphenation validity 100 % |
| 1.2.6 | Overflow ladder v2 incl. "shorter rendering" request (≤ 2 rounds) and plan-then-render (feasibility before erasure), `needs_review.json` (E-08) | property tests; zero ink outside |
| 1.2.7 | Styling: supersampling, contrast-aware fill/outline, polarity, rotated layout, per-type styles | golden-image tests with tolerance |
| 1.2.8 | `TypesetReport` per region (font, size, lines, ink mask, safe polygon, flags) | schema tests; consumed by the gate |

### 1.5 Integration, then 1.4 Gate

| ID | Goal | Acceptance |
|---|---|---|
| 1.5.1 | Feature flags + profiles (`legacy`, `v2`, one per upgrade for ablations); per-region plan-then-render | e2e on dev; legacy reproduces v0.1.0 |
| 1.5.2 | Ablation matrix, performance profile vs `budgets.yaml`, docs | generated tables |
| 1.4.1 | `tools/validate_phase1.py` + tests with negative controls (legacy inpainter swapped in, injected overflow, corrupted shaping must FAIL) | negative controls fail |
| 1.4.2 | Gate run on the sealed test split → `phase1_gate.json` + `quality_report_v2.md`; logged in `gate_runs.log` | exit 0 → human checkpoint; FAIL → `GATE_FAILURE_ANALYSIS.md` |

Edge cases E-01…E-18 each get a test named with the ID (`test_e01_…`), tracked in
`docs/EDGE_CASE_MATRIX.md` and enforced by `tools/check_edge_matrix.py`.

## Known blockers carried into the plan
- G-TR measurement needs NLLB/GGUF weights and a meaningful baseline (ADR-0001 #7, #9).
- The LPIPS part of G-INP-2 needs torchvision backbone weights (ADR-0001 #8).
- Human-verified suites and real gold data depend on the consolidated request.

These are reported as NOT-MEASURABLE / FAIL by the gate until resolved; they are never
passed by construction (PD-3).
