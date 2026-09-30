# Backlog (out of scope for the current phase; PD-6)

Items noticed during work that are not part of the current phase's scope. Each entry:
date, origin (step), description, reason deferred.

| Date | Origin | Item | Why deferred |
|---|---|---|---|
| 2026-09-30 | 0.1 audit | PDF input support | Not supported in v0.1.0; not required by the v0.2.0 gate (ADR-0001 #2) |
| 2026-09-30 | 0.1 audit | `rerender` re-embeds the source ICC profile (store it in the sidecar) | v0.1.0 known defect; not gate-relevant |
| 2026-09-30 | 0.1 audit | ♪ from Noto Sans Symbols renders small (per-font scale for symbol runs) | cosmetic; revisit with the font registry in 1.2 if cheap |
| 2026-09-30 | 0.1 audit | PyInstaller one-folder bundle (D-045) | optional; cannot be verified on this host |
