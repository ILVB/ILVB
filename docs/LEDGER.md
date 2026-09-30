# LEDGER — single source of truth for resuming work (OP-8)

Every step appends one entry: step ID, commit SHA(s), exact commands, key output lines,
decisions, open risks. Entries are append-only; corrections are new entries.

Program: MangaAR v0.2.0 (Phase 0 → Phase 1 → Gate 1 → human checkpoint) → v0.3.0.
Baseline: tag `v0.1.0` = `7ac67181e3aa44cc85ba8efea75ee43a998e15f5`.
Working branch: local `release/0.2.0`, pushed to `origin/claude/peaceful-meitner-4wcz28`
(ADR-0001 #5) until the human authorises a remote `release/0.2.0`.

PD-1 status: **Phase 2 is locked.** Gate 1 has not run; no `PHASE 1 APPROVED` received.

---

## 2026-09-30 — Session start
- Re-read the master prompt (v0.2.0/v0.3.0). Confirmed PD-1 as an absolute HARD GATE.
- Container had restarted; `.venv` and `.cache` (models) persisted.

## Step 0.1 — OP-1 audit — commit `16d20c5`
Commands and key output:
```
$ git rev-parse v0.1.0^{commit} HEAD
7ac67181e3aa44cc85ba8efea75ee43a998e15f5 (both)
$ .venv/bin/python -m pytest -q          # warm
296 passed, 18 deselected in 30.24s
$ .venv/bin/python -m pytest -q -m "integration or slow"
13 passed, 2 skipped, 299 deselected in 204.24s (0:03:24)
$ curl … huggingface.co / download.pytorch.org → 000 ; PyPI, raw.githubusercontent.com → 206
```
Decisions:
- `v0.1.0` exists, so no `v0.1.0-baseline` tag.
- Local branch `release/0.2.0` created from `v0.1.0`.
- ADR-0001 records 12 differences from the prompt.

Open risks:
- NLLB-200, GGUF LLMs and sentence-transformers models are unobtainable here
  (huggingface.co blocked), so G-TR cannot be measured without human action.
- LPIPS backbones are unobtainable here (download.pytorch.org blocked).
- The v0.1.0 translation baseline is TM-only on this host (ADR-0001 #9).

## Step 0.1.2 — Phase 1 plan — commits `8af685e` (templates), `e86ce60` (plan)
- Gold-data templates + JSON Schema in `benchmarks/data/templates/`; consolidated data
  request and six human decisions sent once (chat message, 2026-09-30).
- `docs/PLAN_PHASE1.md` (109 lines) mirrored as session tasks #13–#19.

## Step 0.2.1 — baseline worktree — commit `98f28bd`
```
$ python tools/baseline_worktree.py
baseline OK: /home/user/ILVB/.baseline at v0.1.0 (7ac67181e3aa)
$ git worktree list
/home/user/ILVB/.baseline  7ac6718 (detached HEAD)
$ pytest tests/tools → 4 passed (incl. negative controls: moved tag, dirty/moved worktree)
```
Decision: the lock is JSON (`benchmarks/baseline.lock.json`), because `tomllib` does not
exist on Python 3.10, which the project still supports.

## Step 0.2.2 — common adapter — commits `0b0bfd3`, `2227311`
```
$ pytest -m integration tests/bench/test_adapters_real.py
1 passed in 6.42s
$ pytest -q
305 passed, 19 deselected in 29.95s
```
Decisions:
- Stage-level API (`detect_ocr`, `erase`, `typeset_gt`, `translate_gt`). The harness
  gives both versions identical inputs, including ground-truth geometry and identical
  Arabic strings for typesetting, and the known page language (`--source`).
- One worker subprocess per implementation; `.baseline/src` is placed first on
  PYTHONPATH and verified through `code_origin`.

Open risk: `typeset_gt` rebuilds v0.1.0 `Region`s from ground truth; the type mapping
(thought→bubble, sign/credit→free_text) is recorded in `benchmarks/adapters/v010.py`.

## Step 0.3.2 — synthetic generator — commits `47a7bfc`, `a358514`, `56bc9f4`
```
$ pytest tests/bench/test_generator.py
13 passed in 7.10s   (determinism, GT consistency per category, kinsoku, split disjointness)
```
Decisions:
- The generator is self-contained (never imports `manga_ar`).
- Fonts: 9 OFL fonts, pinned by SHA-256.
- Text bank: 96 parallel meanings × ja/zh/ko/en. Meaning-level splits by keyed hash
  rank (32/32/32).
- Visual inspection of dev pages found CJK lines starting with 。; fixed with kinsoku
  wrapping before freezing.

## Step 0.3.4 — SILVER references — commit `25349c8` (done before 0.3.3)
- Ordering deviation from the plan: references are part of the ground-truth files that
  0.3.3 hashes, so they had to exist first.
- ADR-0002: references authored by the operating model, which is not a system under
  test; always labelled `SILVER-REFERENCE, comparative only`.

## Step 0.3.3 — dataset + manifest + sealed test — commit `0a9d98c`
```
$ python -m benchmarks.generators.build --write-manifest
synthetic_v1: (pages, regions) per split {'dev': (40, 249), 'val': (40, 238), 'test': (50, 300)}; total 787
$ python -m benchmarks.generators.build          # second build → identical manifest (exit 0)
$ python -m benchmarks.generators.build --check
OK
```
- Test split sha256 (aggregate) recorded in `benchmarks/manifests/synthetic_v1.test.sha256`.
- Test files were written and hashed without being read back or printed.
- `load_split("test")` requires the gate's `SealedCapability`; negative controls are
  tested.

## Step 0.3.5 — gold ingestion — commit `03435e3`
```
$ pytest tests/tools/test_ingest_gold.py → 3 passed
$ python tools/ingest_gold.py --check → exit 1 (gold data not delivered yet)
```
