# Status

## Phase checklist
- [x] P0 Preflight, scaffold, spikes (gate passed; tag phase-0-complete)
- [x] P1 Foundation & core contracts (gate passed; tag phase-1-complete)
- [x] P2 Detection, bubbles, OCR (gate passed; tag phase-2-complete)
- [ ] P3 Inpainting
- [ ] P4 Translation
- [ ] P5 Arabic typesetting + ARVS
- [ ] P6 Pipeline, CLI, GUI
- [ ] P7 Hardening, packaging, release

## Known issues
- Network policy blocks HF/Google/MyMemory/Paddle hosts (waivers W-001..W-004).

## Exact next step
Phase 3: inpaint/ (base, mask, solid_fill, opencv_inpaint, lama, strategy, residual) +
invariant tests (outside-mask bit identity, outline band, uniformity, fallback chain).
