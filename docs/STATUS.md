# Status

## Phase checklist
- [x] P0 Preflight, scaffold, spikes (gate passed; tag phase-0-complete)
- [x] P1 Foundation & core contracts (gate passed; tag phase-1-complete)
- [x] P2 Detection, bubbles, OCR (gate passed; tag phase-2-complete)
- [x] P3 Inpainting (gate passed; tag phase-3-complete)
- [x] P4 Translation (gate passed with waivers W-003/W-004; tag phase-4-complete)
- [x] P5 Arabic typesetting + ARVS (gate passed; tag phase-5-complete)
- [x] P6 Pipeline, CLI, GUI (gate passed with waiver W-004; tag phase-6-complete)
- [ ] P7 Hardening, packaging, release

## Phase 5 summary
- ARVS: L1 (fonts/coverage), L2 shaping oracle, L3 bidi, L4 codepoint coverage + symbol
  fallback, L5 joining/direction with negative controls, L6 RAQM oracle overlap (unit);
  L7 OCR round-trip 246/250 samples ≥ 0.8 across 10 fonts (integration, D-031);
  L8 specimen sheets via `scripts/make_specimen_sheet.py` (inspected: RTL, joined,
  lam-alef, mirrored brackets, LTR digits, symbol fallback, text inside padded shapes).
- typeset/ coverage 93 %; typical region ≤ ~45 ms.
- Page composition with untranslated restore + per-region failure isolation (D-034).

## Phase 6 summary
- `pipeline.py`: orchestration with region/page isolation, language vote, reading order,
  SFX policy, sidecar + work images, resume (hash + sha, degraded retry), CBZ repack,
  debug artifacts, cancellation, `rerender`; `report.py` (report.json/.md, exit codes).
- CLI: translate / rerender / demo / gui / doctor / models / fonts (all flags from the spec).
- GUI (Gradio 6): Translate (files, folder, CBZ; progress; cancel; before/after slider;
  ZIP), Review & Edit (region table → overrides → re-render; re-translate region),
  Settings, Diagnostics. Single queue worker, temp workspace cleaned at exit.
- Tests: fake-stage E2E (15), GUI handlers (4) + gradio_client smoke (upload → translate →
  edit → re-render), in-process CLI (4), real-stage CLI E2E (`-m slow`, 4: corrupted batch
  exit 2 + resume + rerender, offline/E4, demo, missing input). Default suite 288 tests
  in ~45 s; overall coverage 87 %.

## Known issues
- Glyphs drawn directly on dense hatching without a halo may be missed (D-019).
- Network policy blocks HF/Google/MyMemory/Paddle hosts (waivers W-001..W-004).
- ♪ from Noto Sans Symbols renders visibly smaller than Arabic text (font metrics).
- `rerender` does not re-embed the source ICC profile (it is not stored in the sidecar).

## Exact next step
Phase 7: fuzzing + 50-page soak, profiling, docs (README with responsible use,
ARCHITECTURE, USER_GUIDE, TROUBLESHOOTING, THIRD_PARTY_LICENSES), launchers, CHANGELOG,
clean-install test, full QUALITY_REPORT (with OCR), non-UTF-8 locale run, v0.1.0.
