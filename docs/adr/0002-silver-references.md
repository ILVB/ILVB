# ADR-0002: SILVER Arabic references for the synthetic benchmark

- Status: accepted (to be superseded by human GOLD references when delivered)
- Date: 2026-09-30
- Phase/Step: 0 / 0.3.4

## Context
G-TR-1 needs Arabic references. The master prompt forbids fabricated references. It
allows "silver" references produced by a model that is NOT under test, with every score
labelled "SILVER-REFERENCE, comparative only". Human GOLD references were requested once
(consolidated request, 2026-09-30) and have not arrived.

## Options
1. Wait for gold references before building the dataset. This blocks the baseline freeze
   (0.2.4) and every translation measurement.
2. Machine-translate the sources with a local MT model to create references. Rejected:
   no MT model is available here (huggingface.co blocked), and any candidate provider
   would then be under test against itself.
3. References authored by the model operating this build, which is not a system under
   test (neither v0.1.0 nor the v0.2.0 providers). Two MSA variants per meaning cover
   addressee gender and number.

## Decision
Option 3: `benchmarks/data/text/refs_ar_silver.yaml` (96 meanings, ≥ 2 references each,
per-language overrides where names differ, e.g. Chinese 小花 → شياو هوا). The file
declares `kind: silver`, and every region carries `reference_kind: silver`.

## Consequences
- Translation scores on the synthetic set are **comparative only**. The gate and reports
  print the `SILVER-REFERENCE, comparative only` label on every translation metric.
- When human GOLD references arrive, they are ingested with `reference_kind: gold` and
  reported separately. Gold always takes precedence.
- Risk: silver references share the author's style. Mitigation: two variants per meaning,
  and human review is requested (the review sheet is generated in 1.1.10).
