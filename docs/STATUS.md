# Status

## Phase checklist
- [x] P0 Preflight, scaffold, spikes (gate passed; tag phase-0-complete)
- [x] P1 Foundation & core contracts (gate passed; tag phase-1-complete)
- [ ] P2 Detection, bubbles, OCR
- [ ] P3 Inpainting
- [ ] P4 Translation
- [ ] P5 Arabic typesetting + ARVS
- [ ] P6 Pipeline, CLI, GUI
- [ ] P7 Hardening, packaging, release

## Known issues
- Network policy blocks HF/Google/MyMemory/Paddle hosts (waivers W-001..W-004).

## Exact next step
Phase 2: detect/ (base, classical, bubble, reading_order, rapid/craft/ctd adapters),
ocr/ (base, engines, router, langid, postprocess, reflow), benchmark metrics + tests.
