# ADR-0003: Performance budgets (step 0.6, G-PERF-1)

Status: accepted (Phase 0); values are open to the human's review at the checkpoint.

## Context
G-PERF-1 requires per-stage p50/p95 latency and peak RAM/VRAM within
`benchmarks/budgets.yaml`, which must be committed after measuring the baseline on dev/val,
before any Phase 1 implementation, and never changed afterwards (PD-3). Phase 1 adds
heavier components (segmentation masks, better recognisers, a quality-driven inpainting
router, local NMT/LLM translation, HarfBuzz shaping and supersampled rendering), so the
budgets must leave room for quality work without allowing unbounded slow-downs.

Measurements: `benchmarks/results/baseline_v0.1.0.json` (SHA-256 `8ab912fe…`), dev + val
(80 pages, 1000×1400), profile `v010-offline`, on the gate host (4 logical cores,
15.7 GB RAM, no GPU; `benchmarks/env.json`). The machine was otherwise idle during the freeze.

## Decision
- **Image stages** (detect, ocr, inpaint, typeset), per page:
  ceiling = max(2 × baseline, baseline + 1.0 s), applied separately to p50 and p95.
  The factor 2 bounds relative slow-down where the stage is already expensive (inpaint
  p95 3.14 s → 6.28 s). The +1 s floor leaves room where the baseline is almost free (inpaint
  p50 0.025 s is mostly solid fills), so a better router can use LaMa more often.
- **Translate**: the offline baseline has no usable provider on this host, so it
  measures ≈ 0 s (ADR-0001 #9) and cannot anchor a relative budget. The absolute ceilings
  are p50 20 s and p95 45 s per page. That is a 20-page chapter in about 7 min median on 4 CPU
  cores, which covers CTranslate2 NLLB and a quantised local LLM for one page's bubbles.
- **Peak RSS**: 8192 MiB (baseline 1452 MiB). That is about half the host RAM, and it covers a
  quantised 7B GGUF model (about 4.5–5 GB) next to the existing OCR and LaMa footprint.
- **Peak VRAM**: not measurable on the gate host (no GPU); recorded as null.
- A stage the gate does not measure counts as a violation, never a silent pass
  (`benchmarks/budgets.py`).
- Immutability: `tools/audit_phase_order.py` fails if any commit after the creating one
  touches `benchmarks/budgets.yaml`.

## Consequences
- Components that break these ceilings must be optimised, made opt-in, or covered by a
  written human waiver shown in the report (G-PERF-1).
- Budgets apply to this host class. On a different gate host the human must decide
  whether to re-measure; that is a change only the human may make.
