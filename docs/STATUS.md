# Status

## Phase checklist
- [x] P0 Preflight, scaffold, spikes (gate passed; tag phase-0-complete)
- [ ] P1 Foundation & core contracts
- [ ] P2 Detection, bubbles, OCR
- [ ] P3 Inpainting
- [ ] P4 Translation
- [ ] P5 Arabic typesetting + ARVS
- [ ] P6 Pipeline, CLI, GUI
- [ ] P7 Hardening, packaging, release

## Known issues
- Network policy blocks HF/Google/MyMemory/Paddle hosts (waivers W-001..W-004).

## Exact next step
Phase 1: write errors.py, config.py (+configs/default.yaml), logging_setup.py, schemas.py,
io/*, models/*, tests/fixtures/synth.py, then unit tests; lock deps (uv lock).
