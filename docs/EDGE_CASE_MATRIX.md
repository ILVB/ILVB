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
| E-01 | SFX / onomatopoeia: `sfx_policy` (keep_original default: not erased, not translated, counted); never in the context window or memory | 1.0.7, 1.1.9 | `tests/edge/test_edge_pipeline_translation.py::test_e01_sfx_is_kept_original_not_translated_and_counted` | covered (v0.1.0 `sfx: skip` = keep_original; the policy set arrives in 1.0.7) |
| E-02 | Text directly on artwork: LaMa route, contrast-aware outline, safe region = dilated text hull | 1.3.2, 1.2.3 | `tests/edge/test_edge_typeset_io.py::test_e02_text_on_artwork_gets_an_outline` | covered (outline over art; LaMa route and dilated-hull safe region in 1.3.2/1.2.3) |
| E-03 | Adjacent / overlapping / connected bubbles, bubbles across panel borders: separated, never merged, stable order, tails excluded from the safe region | 1.2.3 | `tests/edge/test_edge_typeset_io.py::test_e03_touching_bubbles_stay_separate` | covered (separation + stable order; tails excluded in 1.2.3) |
| E-04 | Vertical, rotated or curved source text: detected and recognised; typeset horizontally | 1.0.5 | `tests/edge/test_edge_typeset_io.py::test_e04_vertical_source_is_typeset_horizontally` | covered (horizontal typesetting; recognition quality measured by G-OCR) |
| E-05 | Empty, garbage or low-confidence OCR; punctuation-only bubbles: not translated, not erased, logged | 1.0.6 | `tests/edge/test_edge_pipeline_translation.py::test_e05_punctuation_only_bubble_keeps_original_glyphs`<br>`tests/edge/test_edge_pipeline_translation.py::test_e05_low_confidence_and_garbage_are_never_sent_for_translation` | covered |
| E-06 | Mixed scripts, symbols, ♡★♪: preserved; font fallback; no tofu | 1.2.2 | `tests/edge/test_edge_typeset_io.py::test_e06_symbols_survive_and_never_render_as_tofu` | covered |
| E-07 | Sentences spanning bubbles: context used, output per bubble | 1.1.9 | `tests/edge/test_edge_pipeline_translation.py::test_e07_sentence_spanning_bubbles_translated_in_context_output_per_bubble` | covered (page-level batch context; LLM context window in 1.1.9) |
| E-08 | Translation too long: overflow ladder, never overflows, flagged | 1.2.6 | `tests/edge/test_edge_typeset_io.py::test_e08_too_long_text_is_flagged_never_clipped` | covered (v0.1.0 ladder; ladder v2 in 1.2.6) |
| E-09 | Very short text ("!", "Oh"): size capped relative to the original text height | 1.2.4 |  | untested: v0.1.0 caps size by page fraction, not by the original text height (1.2.4) |
| E-10 | Colour/gradient/screentone backgrounds; 16-bit, CMYK, palette, alpha; EXIF rotation; tall webtoon strips | 1.3.4 | `tests/edge/test_edge_typeset_io.py::test_e10_image_modes_become_rgb_uint8_and_tall_strips_tile` | covered (loader modes + tiling trigger; end-to-end mode preservation in 1.3.4) |
| E-11 | No GPU, OOM, missing model, offline: CPU fallback, explicit warnings, verified cache, actionable failure | 1.3.6 | `tests/edge/test_edge_typeset_io.py::test_e11_oom_falls_back_to_cpu_and_offline_missing_model_is_actionable` | covered |
| E-12 | LLM refusal, invalid JSON, hallucinated additions, meta-commentary, loops, wrong script: filtered, retried, never rendered | 1.1.5, 1.1.9 | `tests/edge/test_edge_pipeline_translation.py::test_e12_wrong_script_and_hallucinated_length_are_never_accepted` | covered (partial: wrong script and hallucinated length; LLM refusal/JSON/meta in 1.1.5) |
| E-13 | Prompt injection inside bubble text: treated as data; injection suite passes | 1.1.9 |  | untested: no LLM prompt exists in v0.1.0 (injection suite in 1.1.9) |
| E-14 | Honorifics, cultural terms, wordplay, names: policy + glossary locks; canonical transliteration | 1.1.6 | `tests/edge/test_edge_pipeline_translation.py::test_e14_glossary_locks_a_name_through_translation` | covered (partial: glossary lock; honorific policy and transliteration in 1.1.6) |
| E-15 | Numbers, dates, units: `digits` western / arabic_indic; consistent; correct mixed direction | 1.2.2 | `tests/edge/test_edge_typeset_io.py::test_e15_digit_policy_and_mixed_direction` | covered |
| E-16 | Reading direction RTL/LTR: configurable, inferred where possible | 1.0.8 | `tests/edge/test_edge_typeset_io.py::test_e16_reading_direction_rtl_vs_ltr` | covered (configurable/default by language; metadata inference in 1.0.8) |
| E-17 | Series memory: isolation, reset/rollback, corrupted-store recovery, single writer | 1.1.6, 1.1.7 |  | untested: series memory does not exist in v0.1.0 (1.1.6/1.1.7) |
| E-18 | Dark bubbles and outlined text: inverted polarity masks; fill colour by contrast | 1.3.1, 1.2.7 | `tests/edge/test_edge_typeset_io.py::test_e18_dark_bubble_gets_light_text` | covered (partial: fill-contrast text colour; mask polarity in 1.3.1) |
