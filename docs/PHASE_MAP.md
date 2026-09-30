# Phase map

Flow: Phase 0 (audit, baseline, harness, governance) → Phase 1 (core engine → v0.2.0) →
Gate 1 (automated) → human checkpoint (`PHASE 1 APPROVED`) → Phase 2 (Edu-Reader →
v0.3.0) → Gate 2 → release.

The path sets and dependency rules below live in `tools/phase_map.py`, the single source
that `tools/audit_phase_order.py` and `tools/assert_gate.py` enforce. Change one only
together with this document.

## Phase 0/1: may change
Detection/OCR, translation, typesetting and inpainting (`src/manga_ar/{detect,ocr,translate,
typeset,inpaint}/`), pipeline orchestration and config (`src/manga_ar/pipeline.py`,
`config.py`, `configs/`), CLI (`src/manga_ar/cli.py`), packaging (`pyproject.toml`,
`uv.lock`), `benchmarks/`, `tests/`, `tools/`, `docs/`, `scripts/`.

The existing Gradio GUI (`src/manga_ar/ui/`) may be touched only for compatibility with
changed internal APIs, in commits prefixed `[compat]` that add no UI feature or behaviour.

## Phase 2 paths: must not exist before tag `phase1-gate-passed`
| Path | Purpose |
|---|---|
| `src/manga_ar/reader/` | reader server, viewer and its static assets |
| `src/manga_ar/edu/` | NLP/dictionary core and Edu-Card |
| `data/` | Phase 2 lexicons (e.g. `data/sfx_lexicon_en_ar.tsv`) |
| `web/` | standalone viewer assets, if split out |
| `tests/edu/`, `tests/reader/` | Phase 2 tests |
| `docs/PLAN_PHASE2.md` | Phase 2 plan (written at step 2.0) |
| `[edu]` extra in `pyproject.toml` | Phase 2 dependencies |

## Dependencies forbidden before the gate
- Anywhere (pyproject.toml or resolved in uv.lock): `spacy`, `spacy-*`, `en-core-web-*`,
  `wordfreq`, `lemminflect`, `pyinflect`, `wn`.
- Declared directly in pyproject.toml: `fastapi`, `uvicorn`, `starlette` (the reader
  server stack). Gradio already pulls them in transitively; that is allowed.
- `nltk` only in the `[bench]` extra (METEOR).

## Phase-1-scoped files (their change invalidates an approved Gate 1)
`src/manga_ar/**`, `benchmarks/**`, `tools/validate_phase1.py`, `configs/**`, except the
Phase 2 paths, `src/manga_ar/ui/**`, `benchmarks/results/**` and `benchmarks/cache/**`.
Dependency changes (pyproject.toml, uv.lock) are not scoped, so Phase 2 can add the
`[edu]` extra, but they are caught by re-running the gate before merging (G2-REG-1).

## Enforcement
| Tool | Where | Fails when |
|---|---|---|
| `python -m tools.audit_phase_order` | CI, pre-commit | a commit after the baseline lacks valid `Phase`/`Step` git trailers; before the gate tag: a `Phase: 2` commit, a Phase 2 path, or a forbidden dependency |
| `python -m tools.audit_phase_order --staged` | pre-commit | the index adds a Phase 2 path or dependency before the gate tag |
| `python -m tools.audit_phase_order --commit-msg F` | commit-msg hook | the message lacks valid trailers, or is `Phase: 2` before the gate tag |
| `python -m tools.assert_gate` | CI (Phase 2), Phase 2 step 2.0 | no valid `.gates/phase1.approved` for a PASS `phase1_gate.json` at an ancestor commit, or a Phase-1-scoped file changed since |
| `python -m tools.assert_gate --if-phase2-staged` | pre-commit | as above, only when the index touches a Phase 2 path |

History starts at the frozen baseline commit (`benchmarks/baseline.lock.json`), because the
remote does not accept tags (ADR-0001 #6). Commits whose `Phase`/`Step` lines sit outside
the git trailer block are listed in `tools/trailer_exceptions.json`. The audit verifies
each against its message and warns about it on every run.
