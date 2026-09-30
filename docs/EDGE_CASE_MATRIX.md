# Edge-case matrix (E-01 … E-18)

Every ID needs at least one automated test whose name contains it (`test_e05_…`).
`python -m tools.check_edge_matrix` verifies this table against the test suite (CI job
`governance`): listed tests must exist, every ID-named test must be listed, and the
status must read `covered` exactly when a test exists. Phase 2 IDs (T-01 … T-17) are
added after the human checkpoint.

"Owner" is the plan step (docs/PLAN_PHASE1.md) that delivers the full behaviour. Tests
written in Phase 0 pin the behaviour v0.1.0 already has; Phase 1 extends them.

| ID | Edge case | Owner | Tests | Status |
|---|---|---|---|---|
| E-01 | SFX / onomatopoeia: `sfx_policy` (keep_original default: not erased, not translated, counted); never in the context window or memory | 1.0.7, 1.1.9 | | untested |
| E-02 | Text directly on artwork: LaMa route, contrast-aware outline, safe region = dilated text hull | 1.3.2, 1.2.3 | | untested |
| E-03 | Adjacent / overlapping / connected bubbles, bubbles across panel borders: separated, never merged, stable order, tails excluded from the safe region | 1.2.3 | | untested |
| E-04 | Vertical, rotated or curved source text: detected and recognised; typeset horizontally | 1.0.5 | | untested |
| E-05 | Empty, garbage or low-confidence OCR; punctuation-only bubbles: not translated, not erased, logged | 1.0.6 | | untested |
| E-06 | Mixed scripts, symbols, ♡★♪: preserved; font fallback; no tofu | 1.2.2 | | untested |
| E-07 | Sentences spanning bubbles: context used, output per bubble | 1.1.9 | | untested |
| E-08 | Translation too long: overflow ladder, never overflows, flagged | 1.2.6 | | untested |
| E-09 | Very short text ("!", "Oh"): size capped relative to the original text height | 1.2.4 | | untested |
| E-10 | Colour/gradient/screentone backgrounds; 16-bit, CMYK, palette, alpha; EXIF rotation; tall webtoon strips | 1.3.4 | | untested |
| E-11 | No GPU, OOM, missing model, offline: CPU fallback, explicit warnings, verified cache, actionable failure | 1.3.6 | | untested |
| E-12 | LLM refusal, invalid JSON, hallucinated additions, meta-commentary, loops, wrong script: filtered, retried, never rendered | 1.1.5, 1.1.9 | | untested |
| E-13 | Prompt injection inside bubble text: treated as data; injection suite passes | 1.1.9 | | untested |
| E-14 | Honorifics, cultural terms, wordplay, names: policy + glossary locks; canonical transliteration | 1.1.6 | | untested |
| E-15 | Numbers, dates, units: `digits` western / arabic_indic; consistent; correct mixed direction | 1.2.2 | | untested |
| E-16 | Reading direction RTL/LTR: configurable, inferred where possible | 1.0.8 | | untested |
| E-17 | Series memory: isolation, reset/rollback, corrupted-store recovery, single writer | 1.1.6, 1.1.7 | | untested |
| E-18 | Dark bubbles and outlined text: inverted polarity masks; fill colour by contrast | 1.3.1, 1.2.7 | | untested |
