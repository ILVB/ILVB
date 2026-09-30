# Status

## Phase checklist
- [x] P0 Preflight, scaffold, spikes (gate passed; tag phase-0-complete)
- [x] P1 Foundation & core contracts (gate passed; tag phase-1-complete)
- [x] P2 Detection, bubbles, OCR (gate passed; tag phase-2-complete)
- [x] P3 Inpainting (gate passed; tag phase-3-complete)
- [x] P4 Translation (gate passed with waivers W-003/W-004; tag phase-4-complete)
- [ ] P5 Arabic typesetting + ARVS
- [ ] P6 Pipeline, CLI, GUI
- [ ] P7 Hardening, packaging, release

## Known issues
- Glyphs drawn directly on dense hatching without a halo may be missed (D-019).
- Network policy blocks HF/Google/MyMemory/Paddle hosts (waivers W-001..W-004).

## Exact next step
Phase 5: typeset/ (arabic_text shim with L2/L4 bidi, wrap, layout A/B, fit, overflow ladder,
contrast, render) + ARVS L1–L8 + specimen sheet.
