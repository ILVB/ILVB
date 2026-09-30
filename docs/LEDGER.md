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
