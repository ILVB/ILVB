# Status

## v0.2.0 program (branch `release/0.2.0`, pushed to `claude/peaceful-meitner-4wcz28`)
Ledger: `docs/LEDGER.md`. Plan: `docs/PLAN_PHASE1.md`. Phase map: `docs/PHASE_MAP.md`.

- [x] Phase 0: audit (0.1), baseline worktree + adapter (0.2.1–0.2.2), metrics + harness
  (0.2.3), frozen baseline for dev/val/test (0.2.4), synthetic_v1 dataset + sealed test
  (0.3), environment snapshot (0.4), governance (0.5), budgets (0.6, ADR-0003)
- [ ] Phase 1: core engine → v0.2.0, in progress. Done: 1.0.1 TextRegion + `engine.profile`,
  1.0.2 per-category baseline, 1.0.3 v2 detector (db_primary, ADR-0004). Done in part:
  1.1.3 defences (E-12/E-13), 1.1.6 Series Bible store (E-17), 1.2.4 size ceiling (E-09).
  Edge matrix 18/18. The legacy profile reproduces v0.1.0 exactly on dev.
- [ ] Gate 1 → human checkpoint (`PHASE 1 APPROVED`) — Phase 2 locked (PD-1)

Human decisions: ADR-0007 (2026-09-30). Waiting on the human's host: running
`scripts/download_heavy_weights.py` (W-HEAVY-1 waiver until then) and
`python -m tools.freeze_online_baseline` (the v0.1.0_online_baseline for G-TR). Real gold
data is still pending.

Next step: 1.0.4 preprocessing and 1.0.5 recognizer selection (the remaining CER is
recognition: stylized, text_on_art, tiny_text), then 1.1 providers and router.

## v0.1.0 phase checklist
- [x] P0 Preflight, scaffold, spikes (gate passed; tag phase-0-complete)
- [x] P1 Foundation & core contracts (gate passed; tag phase-1-complete)
- [x] P2 Detection, bubbles, OCR (gate passed; tag phase-2-complete)
- [x] P3 Inpainting (gate passed; tag phase-3-complete)
- [x] P4 Translation (gate passed with waivers W-003/W-004; tag phase-4-complete)
- [x] P5 Arabic typesetting + ARVS (gate passed; tag phase-5-complete)
- [x] P6 Pipeline, CLI, GUI (gate passed with waiver W-004; tag phase-6-complete)
- [x] P7 Hardening, packaging, release (gate passed; tag v0.1.0)

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

## Phase 7 summary
- Fuzzing: loader, archive, normalize_ar, shaping/bidi, sidecar. The deep run used 3,000
  examples per property and found one bug (a Pillow TypeError on a corrupt TIFF), now
  fixed.
- Soak: 50 pages with real stages; RSS grew 5 MB after warm-up; 0.85 s/page (fast
  preset, before the performance pass).
- Profiling:
  - detector windowing gives identical output at 2.4× speed;
  - end to end: 0.9 → 0.47 s/page.
- Docs: README (responsible use), USER_GUIDE, ARCHITECTURE, TROUBLESHOOTING (E1–E21 →
  tests), THIRD_PARTY_LICENSES (script-generated), CHANGELOG.
- Packaging: GUI launchers; clean-install test (sh/ps1), which passed: fresh clone →
  doctor → demo 3/3.
- Release blocker found and fixed: `.gitignore` had excluded `src/manga_ar/models/`; a
  repo-hygiene test now guards it.
- Locale: suite green with PYTHONUTF8=1 and under LANG=C, UTF-8 mode off.
- Licence flag: PyPI torch pulls proprietary NVIDIA libraries on Linux (D-043); this is
  a decision for the user.
- Final numbers: default suite 296 passed in about 30 s (48 s with coverage); coverage
  88 % overall, typeset/ 93 %, resilience 97 %.

## Known issues
- Glyphs drawn directly on dense hatching without a halo may be missed (D-019).
- Network policy blocks HF/Google/MyMemory/Paddle hosts (waivers W-001..W-004).
- ♪ from Noto Sans Symbols renders visibly smaller than Arabic text (font metrics).
- `rerender` does not re-embed the source ICC profile (it is not stored in the sidecar).
- No PyInstaller spec (optional; D-045).

## Exact next step
Released v0.1.0. Suggested next steps are listed in the Final Delivery Report:
- verify W-001..W-004 on an unrestricted host;
- decide on CPU-only torch in the lock (D-043);
- stretch items.
