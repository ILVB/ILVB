# Status

## Phase checklist
- [x] P0 Preflight, scaffold, spikes (gate passed; tag phase-0-complete)
- [x] P1 Foundation & core contracts (gate passed; tag phase-1-complete)
- [x] P2 Detection, bubbles, OCR (gate passed; tag phase-2-complete)
- [x] P3 Inpainting (gate passed; tag phase-3-complete)
- [ ] P4 Translation
- [ ] P5 Arabic typesetting + ARVS
- [ ] P6 Pipeline, CLI, GUI
- [ ] P7 Hardening, packaging, release

## Known issues
- Glyphs drawn directly on dense hatching without a halo may be missed (D-019).
- Network policy blocks HF/Google/MyMemory/Paddle hosts (waivers W-001..W-004).

## Exact next step
Phase 4: translate/ (base, providers, resilience, cache, batching, glossary, tm, local_mt,
normalize_ar) + fake-clock resilience suite; network smoke waived (W-003).
