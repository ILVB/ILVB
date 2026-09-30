# Troubleshooting

Always start with:

```bash
manga-arabic doctor          # add --no-network to skip reachability checks
```

It reports versions, the OpenCV distribution, RAQM support, the device, model presence,
font coverage and which translation endpoints are reachable. `out/report.md` explains
per page what happened.

Every edge case below (E1–E21) has at least one automated test, named in *Tests*.
Paths are relative to `tests/`.

---

## Translation

### E1 — Translation times out or the connection drops
**Symptom:** a slow run; the log shows `deadline exceeded` or connection errors.
**What MangaAR does:**
1. Every call has a hard deadline (`translate.total_timeout`, 25 s).
2. The call is retried with exponential backoff and jitter.
3. It then fails over to the next provider, and finally to the local model.
4. If everything fails, the region is flagged `UNTRANSLATED`. The page is still
   exported, with the original text kept in that bubble.

**What you can do:** rerun later with `--resume` (only untranslated regions are
retried), or install offline translation (E4).
*Tests:* `unit/test_translate_resilience.py::test_deadline_wrapper_bounds_hung_calls`,
`::test_deadline_inside_caller_is_retried`,
`unit/test_translate_service.py::test_hung_provider_is_time_bounded`,
`::test_all_providers_fail_flags_untranslated_and_page_continues`.

### E2 — HTTP 429 / quota exhausted
**Symptom:** the log shows `429`, `TooManyRequests` or `circuit open`.
**What MangaAR does:**
- A client-side limiter (1 request/s) paces requests.
- Backoff honours `Retry-After`.
- After 5 consecutive failures the provider's circuit breaker opens for 120 s, and
  traffic moves to the next provider.
- The SQLite cache means repeated text is never requested twice.

**What you can do:** lower `translate.rate_per_sec`, add a translation memory, use
`--providers tm,local`, or point `translate.libretranslate_url` at your own instance.
Never try to bypass a service's limits.
*Tests:* `unit/test_translate_resilience.py::test_429_honours_retry_after`,
`::test_breaker_open_half_open_close`,
`unit/test_translate_service.py::test_breaker_shifts_traffic_to_next_provider`,
`::test_rate_limit_then_success_with_retry_after`,
`::test_cache_hit_miss_and_persistence`.

### E3 — Garbage translation
**Symptom:** a bubble came back empty, unchanged, in the wrong script or scrambled.
**What MangaAR does:** outputs are validated:
- empty, identical to the source, or containing CJK are rejected;
- so are an Arabic-letter ratio below 0.6, an extreme length, or lost glossary tokens;
- a page batch whose `[n]` markers don't survive falls back to per-bubble requests.

A rejected output goes to the next provider; if every provider fails, the region is
flagged `UNTRANSLATED`.
**What you can do:** fix it in **Review & Edit**, or add the phrase to your TM.
*Tests:* `unit/test_translate_service.py::test_wrong_script_rejected_then_next_provider`,
`::test_batch_count_mismatch_falls_back_to_per_region`,
`::test_glossary_placeholders_survive_and_degrade`.

### E4 — Fully offline
**Symptom:** with `--offline` (or no internet) every region is `UNTRANSLATED`, and the
report says no offline model is installed.
**What MangaAR does:** network providers are skipped. Translation comes from the TM, the
local model or the cache. Without them, original pixels are kept and the report says
what to install.
**What you can do:** while online, run
`pip install -e ".[offline-mt]" && manga-arabic models download local-mt`, or provide
`--tm`. Behind a proxy that blocks huggingface.co, download the four OPUS-MT
repositories manually into `<cache>/models/hf/` as the error message describes.
*Tests:* `e2e/test_cli_real.py::test_offline_without_local_model_keeps_originals_and_explains`,
`e2e/test_pipeline_fake.py::test_degraded_page_retries_translation_on_resume`,
`unit/test_translate_service.py::test_offline_skips_network_and_suspect_and_skips`,
`unit/test_local_mt.py`.

## OCR and language

### E5 — Wrong or garbage OCR
**Symptom:** the source text is empty or nonsense, flagged `OCR_SUSPECT`,
`LOW_CONFIDENCE` or `OCR_FAILED`.
**What MangaAR does:** each result is checked:
- empty output, wrong script, repetition loops, implausible density and low confidence
  are all suspicious;
- a suspicious result is retried with the next engine or an upscaled crop;
- the best candidate is kept, and still flagged if nothing is good.

Suspect text is **not** translated (`translate.translate_suspect: false`).
**What you can do:** correct it in the GUI by typing the Arabic yourself, try
`--preset quality` (multi-engine OCR), or install the `manga` extra for Japanese.
*Tests:* `unit/test_ocr_logic.py::test_garbage_is_flagged`,
`::test_suspicious_result_retries_next_engine`, `::test_all_engines_fail_flags_ocr_failed`,
`e2e/test_pipeline_fake.py::test_region_failures_are_isolated`.

### E6 — Wrong source language detected
**Symptom:** Korean read as Chinese, and similar.
**What MangaAR does:** the language is voted per book, over three pages, from each
engine's confidence and script agreement (kana → ja, hangul → ko, han only → zh).
**What you can do:** pass `--source ja|ko|zh`. For a single region, set
`override.source_lang` in the sidecar; `--resume` and **Re-translate region** honour it.
*Tests:* `unit/test_ocr_logic.py::test_language_detection_with_fakes`,
`unit/test_ocr_text.py`,
`e2e/test_pipeline_fake.py::test_region_language_override_on_resume`,
`::test_grayscale_debug_sfx_and_language_vote`.

### E7 — Vertical text on engines that read horizontally
**What MangaAR does:** vertical columns are sliced into glyph cells and reflowed right to
left (gap-aware, so split glyphs like こ stay whole).
*Tests:* `unit/test_ocr_logic.py::test_vertical_region_is_reflowed_for_line_engines`,
`unit/test_ocr_text.py`, `integration/test_ocr_engines.py` (vertical Japanese CER gate).

## Typesetting

### E8 — Text does not fit
**Symptom:** the flag `OVERFLOW_RISK`, or very small text.
**What MangaAR does:** it walks the overflow ladder:
1. tighter line spacing;
2. a condensed width;
3. less padding;
4. growing into surrounding uniform background, never across outlines;
5. the minimum size, flagged.

It never clips and never draws off the page.
**What you can do:** shorten the translation, choose a narrower font (Tajawal, Almarai),
set a size override, or use `typeset.strategy: rect` for odd shapes.
*Tests:* `unit/test_typeset.py::test_overflow_ladder_order_and_flag`,
`::test_property_zero_overflow`, `::test_extend_into_uniform_background_for_free_text`.

### E13 — Missing fonts, glyphs or symbols
**Symptom:** `unknown font`, or a symbol such as ♡ missing from the output.
**What MangaAR does:**
- Every emitted code point is checked against the font's cmap.
- Symbols come from Noto Sans Symbols 1/2.
- A glyph that no font has is dropped and logged, never rendered as tofu.
- Fonts without contextual forms (Baloo Bhaijaan 2) are only offered on the RAQM path.

**What you can do:** `manga-arabic fonts check`; use a bundled font key.
*Tests:* `unit/test_arvs.py::test_l4_every_emitted_codepoint_is_covered`,
`::test_l4_symbol_fallback_engages`.

### Arabic looks disconnected or backwards
This cannot happen with the bundled fonts. ARVS L2–L7 check joining and direction on
every bundled font, with negative controls. If you configured a system font,
`fonts check` must show `BASIC+RAQM`. Never pre-shape or reverse text yourself;
MangaAR's `normalize_ar` removes presentation forms and bidi controls it receives.

### E15 — Text touching the bubble border or overlapping artwork
**What MangaAR does:** inpainting is confined to an allowed zone (the bubble interior
eroded ≥ 2 px), so outlines and tails survive. Lettering keeps a padding from the
outline, and free text over artwork gets a contrasting outline.
*Tests:* `unit/test_inpaint.py::test_bubble_outlines_untouched`,
`::test_allowed_zone_keeps_outline_at_mask_extremes`,
`unit/test_typeset.py::test_padding_erosion_never_reaches_the_outline`,
`::test_text_stays_inside_bubble`.

### E20 — Bidi control or RTL-override characters in text
**What MangaAR does:** `normalize_ar` strips LRM, RLM, LRE, RLE, PDF, LRO, RLO, LRI, RLI,
FSI, PDI and ALM, zero-width characters and tatweel from OCR and translation output.
This prevents spoofing and double reordering.
*Tests:* `unit/test_normalize_ar.py::test_no_bidi_controls_survive`,
`unit/test_fuzz.py::test_normalize_ar_properties`,
`unit/test_typeset.py::test_harakat_stripped_digits_and_bidi_controls`.

## Images, detection and inpainting

### E9 — Corrupted or unsupported images
**Symptom:** a page has status `skipped` in the report.
**What MangaAR does:**
- The following raise a typed `ImageLoadError`: zero-byte files, truncated data (in
  `input.truncated: strict` mode), bad CRCs, non-images, 1×1 images, absurd dimensions
  and decompression bombs.
- The page is skipped and reported, and the batch continues (exit code 2).
- In the default `lenient` mode, truncated files are decoded best-effort, with a warning.
- Handled correctly rather than rejected: animated images (first frame), CMYK, 16-bit,
  palette and alpha images, EXIF rotation, and non-ASCII or very long paths.

**What you can do:** re-export the file from an image editor.
*Tests:* `unit/test_loader.py` (all), `unit/test_fuzz.py::test_loader_never_raises_untyped`,
`e2e/test_pipeline_fake.py::test_batch_with_corrupted_files_exit_2_and_layout`,
`e2e/test_cli_real.py::test_translate_corrupted_batch_exit_2_then_resume`.

### E10 — Huge or very tall images (webtoons)
**What MangaAR does:** strips taller than 2.5× their width (or 4096 px) are tiled
(≤ 2048 px, overlap ≥ 192 px, cut at quiet gutters). Seam duplicates are merged, and
out-of-memory errors retry with halved tiles.
**What you can do:** `--preset fast`, `--device cpu`, or split the strip.
*Tests:* `unit/test_tiling.py` (including `::test_detection_retries_smaller_tiles_on_memory_error`),
`unit/test_detect.py::test_webtoon_seam_each_bubble_once`.

### E11 — Model download fails, checksum mismatch, low disk, first run offline
**Symptom:** `ModelUnavailableError: … To install it manually …`.
**What MangaAR does:**
- Downloads are retried with backoff, resume partial files and verify SHA-256.
- Free disk space is checked before downloading.
- Offline mode never touches the network.
- Detection falls back to the classical detector and inpainting to OpenCV, so the
  pipeline still works.

**What you can do:** follow the manual instructions in the message, or run
`manga-arabic models download default` once while online.
*Tests:* `unit/test_models.py` (all), `integration/test_lama.py`.

### E12 — GPU out of memory, or CUDA/MPS failure
**What MangaAR does:** device errors are caught, the accelerator cache is freed and the
call is retried on the CPU, logged once. `--device auto` falls back to the CPU when no
GPU exists.
**What you can do:** `--device cpu`.
*Tests:* `unit/test_device_logging.py::test_cpu_fallback_on_oom`,
`::test_resolve_device_cpu_fallbacks`.

### E14 — A bubble mask leaks or segmentation fails
**Symptom:** the flag `LEAK_FALLBACK`.
**What MangaAR does:** an implausible region is discarded and replaced by a shape fitted
around the text. Implausible means: too large compared with the text, over 35% of the
page, touching the border, low solidity, or merging with gutters. The replacement is
typeset with the inscribed-rectangle strategy.
*Tests:* `unit/test_bubble.py::test_leak_fallback_on_open_bubble`,
`unit/test_typeset.py::test_leak_fallback_uses_rectangle_strategy`.

### E21 — A page with no detected text
**What MangaAR does:** the page is exported unchanged, with status `no_text`, logged at
INFO.
*Tests:* `e2e/test_pipeline_fake.py::test_erase_untranslated_and_no_text_page`.

### Text is still visible after inpainting
**Symptom:** the flag `RESIDUAL_TEXT`.
**What you can do:** use `--preset quality` (LaMa plus a residual check with larger
masks), or install the `lama` extra. Glyphs drawn straight onto dense hatching may be
missed by the classical detector (D-019); `--detector hybrid` adds a neural detector.

## Batches, archives and resume

### E16 — Archive hazards
**What MangaAR does:**
- Member names are validated: `../` or absolute paths reject the whole archive.
- Archives are read in memory, in natural order.
- Non-images and directories are ignored, and the first of any duplicate names is kept.
- The member count, per-member size and total size are capped (`input.archive_*`).

*Tests:* `unit/test_archive.py` (all), `unit/test_fuzz.py::test_archive_member_names`,
`::test_archive_reader_is_typed`.

### E18 — Partial batch failure
**What MangaAR does:** each page is isolated. The exit code is 0 (all succeeded), 2
(partial) or 1 (nothing processed). `report.md` and `report.json` list every page with
its status and reason.
*Tests:* `e2e/test_pipeline_fake.py::test_page_crash_is_isolated`,
`::test_all_failed_is_fatal_and_missing_input`, `e2e/test_cli_real.py`.

### E19 — Stale results or caches after an upgrade or setting change
**What MangaAR does:** `--resume` compares the page's config hash (which includes the
MangaAR version) and the source image hash, and reprocesses when they differ. The
translation-cache keys include the cache version. `--force` reprocesses everything.
**What you can do:** delete `<cache>/translations.sqlite3` to force fresh translations.
*Tests:* `e2e/test_pipeline_fake.py::test_resume_skips_and_config_change_reprocesses`,
`unit/test_config.py`, `unit/test_translate_service.py::test_cache_hit_miss_and_persistence`.

## GUI

### E17 — GUI concurrency, cancellation and clean-up
**What MangaAR does:**
- One worker processes jobs in order, because ML runtimes are not thread-safe.
- **Cancel** stops after the current region or page; the partial results and report are
  kept.
- Uploads and results live in a temporary workspace that is deleted when the GUI exits,
  and Gradio's own copies expire hourly.

*Tests:* `e2e/test_gui_smoke.py` (queue concurrency 1, upload → translate → edit →
re-render), `unit/test_gui_handlers.py::test_cbz_page_preview_cancel_diagnostics_cleanup`,
`e2e/test_pipeline_fake.py::test_cancellation_between_pages`.

### The GUI does not open, or the port is busy
Use `manga-arabic gui --port 7870`. Use `--no-browser` on headless machines, then open
`http://127.0.0.1:7870` yourself. Binding to another host (`--host 0.0.0.0`) exposes the
GUI to your network and logs a warning.

## Installation

### "two OpenCV distributions installed" (doctor fails)
Some OCR packages pull `opencv-python` next to `opencv-python-headless`, and both provide
`cv2`. Fix it with:

```bash
pip uninstall -y opencv-python opencv-contrib-python
pip install --force-reinstall opencv-python-headless
```

With `uv sync`, the lock file already prevents this.

### "RAQM not available"
Only needed for `typeset.render_path: raqm` and the Baloo Bhaijaan 2 font. The default
path does not need it. Pillow wheels ship libraqm on Linux and macOS; on Windows, install
FriBiDi or keep the default path.

### Arabic or CJK characters look garbled in the Windows console
MangaAR reconfigures stdout/stderr to UTF-8 with replacement. If your terminal still
shows `?`, run `chcp 65001` or use Windows Terminal. The files themselves are always
correct UTF-8.

### Very long paths on Windows
Paths longer than 260 characters are handled with the `\\?\` prefix. Enable long paths
in Windows if other tools fail on the outputs.

### PyTorch is huge
It is only needed by the `ocr`, `lama`, `manga` and `offline-mt` extras. For Chinese-only
use, `pip install -e ".[rapid,gui]"` avoids it. On Linux you can install the CPU wheel
from `https://download.pytorch.org/whl/cpu` to save about 2 GB.
