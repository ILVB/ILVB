# Status

## Phase checklist
- [x] P0 Preflight, scaffold, spikes (gate passed; tag phase-0-complete)
- [x] P1 Foundation & core contracts (gate passed; tag phase-1-complete)
- [x] P2 Detection, bubbles, OCR (gate passed; tag phase-2-complete)
- [x] P3 Inpainting (gate passed; tag phase-3-complete)
- [x] P4 Translation (gate passed with waivers W-003/W-004; tag phase-4-complete)
- [x] P5 Arabic typesetting + ARVS (gate passed; tag phase-5-complete)
- [ ] P6 Pipeline, CLI, GUI
- [ ] P7 Hardening, packaging, release

## Phase 5 summary
- ARVS: L1 (fonts/coverage), L2 shaping oracle, L3 bidi, L4 codepoint coverage + symbol
  fallback, L5 joining/direction with negative controls, L6 RAQM oracle overlap (unit);
  L7 OCR round-trip 246/250 samples ≥ 0.8 across 10 fonts (integration, D-031);
  L8 specimen sheets via `scripts/make_specimen_sheet.py` (inspected: RTL, joined,
  lam-alef, mirrored brackets, LTR digits, symbol fallback, text inside padded shapes).
- typeset/ coverage 93 %; typical region ≤ ~45 ms.
- Page composition with untranslated restore + per-region failure isolation (D-034).

## Known issues
- Glyphs drawn directly on dense hatching without a halo may be missed (D-019).
- Network policy blocks HF/Google/MyMemory/Paddle hosts (waivers W-001..W-004).
- ♪ from Noto Sans Symbols renders visibly smaller than Arabic text (font metrics).
- `detect.sfx: translate` is not wired yet (all stages skip SFX) → Phase 6.

## Exact next step
Phase 6: pipeline.py (orchestration, isolation, resume, export, report), full CLI
(translate/gui/rerender/demo), Gradio GUI (4 tabs), E2E tests.
