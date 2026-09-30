# Decisions Log

Format: **ID — decision.** Why · alternatives · evidence. Newest entries are appended.

## Phase 0

**D-001 — Project root is the repo root.** The directory contained only `.git`, so the
"unrelated files" clause does not apply.

**D-002 — Python 3.11.15 in `.venv` via uv.** It is the host default and has wheels for every
heavy dependency (torch 2.14, onnxruntime 1.30, opencv 5.0, gradio 6.29). 3.10/3.12 are
declared compatible in `pyproject.toml`.

**D-003 — Egress restrictions (see ENV_AUDIT.md) shape what can be verified.** Hugging Face,
Google Translate, MyMemory, Paddle model hosts and all offline-MT model hosts are blocked
by the session's network policy. Implementations for these stay behind interfaces and are
tested with fakes. Live gates are waived (W-xxx entries below) instead of faked.

**D-004 — Arabic fonts are vendored; CJK fixture fonts are downloaded.** Eleven OFL files
(~4.9 MB) sit in `src/manga_ar/assets/fonts/` (see D-006) with OFL texts in `licenses/` and a
`SHA256SUMS`. The Noto CJK subsets (~17 MB) are fetched by `scripts/download_assets.py`
with pinned SHA-256s and never committed.

### Spike verdicts

**SP-A Arabic rendering — PASS (with two required fixes).** `spikes/sp_a_arabic_render.py`
rendered 8 strings × 2 fonts through Path 1 (reshaper + bidi + BASIC) and Path 2 (RAQM).
- Joining oracle (96 px, binarised connected components): `محمد` = 1 and `الحمد` = 2 in
  Noto Naskh and Tajawal, on both paths. The unshaped negative control gives 4 and 5, so
  the check can tell the cases apart.
- Visual inspection: letters are joined, words run right→left, and `!`/`؟` land at the
  visual left. Path 1 and Path 2 match for Noto Naskh.
- **Fix 1: python-bidi 0.6.11 does not mirror brackets.** Its Rust `get_display` does UBA
  reordering, including N0 bracket pairs, but skips rule L4 mirroring. `على (OK)` came out
  as `)OK(`, while RAQM gave `(OK)`. The legacy pure-Python `bidi.algorithm` does mirror
  but reorders `[x] 10` wrongly. Decision: the shim in `typeset/arabic_text.py` takes the
  resolved levels from the Rust engine, applies L2 reordering and L4 mirroring itself, and
  is cross-checked against the Rust visual order and against RAQM.
- **Fix 2: many OFL Arabic fonts lack Presentation Forms-B.** Tajawal, Cairo, Almarai,
  Changa and Lalezar have the initial/medial/final forms but not the *isolated* forms
  (FE8D, FE83, …), so Path 1 drew tofu. Decision: a per-font reshaper configuration emits
  base letters instead of isolated forms (`use_unshaped_instead_of_isolated`) when a font
  lacks them. Emitted codepoints are also checked against the cmap at runtime.
  Baloo Bhaijaan 2 lacks almost all forms, so it is RAQM-only.
- ♡ was tofu in every Arabic font, so a symbol fallback is mandatory. Noto Sans Symbols 2
  covers ♡♥☆★ and Noto Sans Symbols (vendored) covers ♪♫.

**SP-B manga-ocr — FAIL (environment).** `MangaOcr()` raised `OSError` because
huggingface.co is blocked (403). Fallback for Japanese: EasyOCR `ja` (read
`今日はいい天気ですね` exactly, confidence 1.0). The manga-ocr adapter is kept (extra
`manga`); it runs wherever the Hub is reachable. Waiver W-001.

**SP-C PaddleOCR — FAIL on this host → RapidOCR + EasyOCR substitute.** paddleocr 3.7.0 /
paddlepaddle 3.3.1 / paddlex 3.7.2 install in an isolated venv. The API is
`PaddleOCR(lang=…, use_textline_orientation=…)` with `.predict()`. Pipeline creation fails
on missing extras, and every model host (BOS, AIStudio, ModelScope, HF) is blocked. As the
prompt allows, the substitute is **RapidOCR** (`rapidocr_onnxruntime` 1.4.4, PP-OCR ONNX
models bundled in the wheel). It read `今天天气很好` exactly, plus both columns of a
vertical sample. RapidOCR's bundled Chinese model returns nothing on Hangul, so Korean
routes to **EasyOCR** `ko` (1 wrong character out of 9 on the sample). The Paddle adapter
targets the 3.x `predict()` result schema, with a 2.x fallback, but is unverified here.
Waiver W-002.

**SP-D detectors — PASS (classical baseline + optional ML).** Synthetic page, 4
ground-truth blocks (`spikes/out/sp_d_*.png`, inspected):
| detector | boxes | truth matched @IoU0.3 | time |
|---|---|---|---|
| classical (adaptive threshold + CC) | 8 (per column) | 4/4 | 0.03 s |
| RapidOCR DB (bundled, Apache-2.0) | 10 (fragments vertical text) | 3/4 | 0.58 s |
| EasyOCR CRAFT (GitHub weights) | 8 (per column) | 4/4 | 9.4 s (incl. load) |
| comic-text-detector ONNX (GPL-3.0 weights) | 4 (whole blocks) | 4/4 | 35 s (incl. 95 MB download) |
Screentone produced no false positives with any detector. Decision: classical is the
mandatory default. Line boxes are grouped into blocks per bubble. comic-text-detector is
**opt-in only** (`detector: ctd`) because its weights are GPL-3.0. This is flagged to the
human as a licensing decision.

**SP-E inpainting — PASS.** Gradient bubble, mean |error| against the clean gradient
inside the mask: solid 5.02, Telea 1.31, **NS 0.72**, LaMa 3.36. All four left pixels
outside the mask bit-identical. Visual check: solid fill leaves a flat patch on
gradients. Telea and NS are clean. LaMa (big-lama TorchScript, CPU, 2.2 s for 480×360)
leaves faint ghosting on smooth gradients. Decision: the strategy selector classifies the
background as uniform (solid fill), smooth gradient (NS/Telea) or textured (LaMa, falling
back to Telea). LaMa is loaded directly with `torch.jit.load` from a SHA-256-verified file,
so the `simple-lama-inpainting` wrapper (which pulls opencv-python) is not needed.
big-lama sha256 `344c77bb…a9ea9`, 205,669,692 bytes.

**SP-F translation — FAIL (environment); resilience wrapper verified.** Both deep-translator
Google and MyMemory failed fast through the deadline + backoff wrapper (ProxyError /
RequestError, bounded retries). Marian local MT failed with 403 from huggingface.co.
Findings for the implementation:
- deep-translator 1.11.4 passes **no timeout** to `requests.get`, so the executor deadline
  is mandatory.
- Google limit: `len < 5000` characters, so chunk at ≤ 4500.
- MyMemory limit: `len < 500` characters, so chunk at ≤ 450.

Waivers W-003 and W-004.

**SP-G Gradio — PASS.** gradio 6.29.0 / gradio_client 2.7.1: Blocks app bound to
127.0.0.1 with `queue(default_concurrency_limit=1)`. An upload through
`gradio_client.Client.predict(handle_file(...), api_name="/invert")` returned the processed
image. The server made no outbound connections; the client pinged huggingface.co once.

**D-005 — Translation memory (TM) provider.** No translation endpoint or MT model is
reachable here, yet `demo` must show real Arabic output offline. So an exact-match
translation memory (a standard CAT-tool feature: normalised source → Arabic, loaded from
user JSON/YAML/CSV) is the first provider in the chain when configured. The demo ships a
TM for its own synthetic sentences. This is a real, user-facing feature and the report
names the provider used. The pipeline does not special-case the demo.

## Phase 1

**D-006 — Repo layout adaptations.** (a) The vendored fonts live in
`src/manga_ar/assets/fonts/` instead of a top-level `assets/fonts/`, because the wheel
must ship them (`manga-arabic demo` and typesetting need them after `pip install`).
(b) The synthetic page generator lives in `manga_ar/synth.py`, because the `demo`
subcommand needs it at runtime; `tests/fixtures/synth.py` re-exports it and adds
corrupted-file builders. (c) `doctor.py` and `resources.py` are small extra modules.
(d) CJK fixture fonts are registry "models" (`font-cjk-*`), so they get SHA-256
verification and offline handling from the model manager.

**D-007 — Defaults live in packaged `default.yaml`.** Config dataclasses have no field
defaults; every key must come from `manga_ar/data/default.yaml`, so no default can hide in
code. `configs/default.yaml` is a byte-identical copy for readers, enforced by a unit
test.

**D-008 — Config hash scope.** The hash excludes keys that cannot change output pixels or
text (device, debug, cache_dir, log level, threads, offline, resume/force) and includes
the MangaAR version (E19).

**D-009 — Lock file.** `uv lock` resolves 167 packages for Python 3.10–3.12 on all
platforms. The venv is synced with `uv sync --locked` (all extras except paddle/gpu), and
all tests run against the locked versions.

**D-010 — Loader policy details.** Images smaller than 16 px on a side (e.g. 1×1) are
rejected as unusable pages. Pillow's default pixel limit (~179 MP) is the
decompression-bomb guard. Bad PNG CRCs count as corruption: `strict` raises, `lenient`
decodes best-effort with a warning.

## Phase 2

**D-011 — Default detector is the classical one; ML detectors are adapters.** Seeds
10–11, ja/ko/zh basic + variety pages, P/R at IoU 0.5:
| detector | precision | recall |
|---|---|---|
| classical | 0.977 → 1.000 after the polarity fix | 0.977 → 1.000 after the polarity fix |
| rapid (PP-OCR DB) | 0.953 | 0.953 |
| craft (EasyOCR) | 0.909 | 0.930 |
| ctd (GPL-3.0 weights) | 1.000 | 0.860 (block boxes looser than GT) |
`detect.detector: auto` → classical: no download, best score. `rapid`, `craft` and `ctd`
are selectable and fall back to classical if unavailable.

**D-012 — Classical detector design (calibrated on seeds 0–2, verified on 10–15, 20–21).**
- Glyph candidates are connected components passing size, fill and contrasting-surround
  tests.
- Thin diagonal strokes are hatching. Axis-aligned thin strokes (一, ー) are kept.
- Grouping follows the enclosing light region, then proximity.
- Polarity conflicts are settled by a ring-uniformity test with symmetric
  size-comparability. This handles white text in black bubbles and glyph counters.
- Nested blocks inside large-glyph blocks are absorbed, and small groups on textured
  backgrounds are rejected.
- Orientation: each glyph's nearest neighbour lies along the reading direction. Glyphs
  are assembled from pieces (≤ 1.15 glyph, square-ish) before voting.
- Furigana are split off columns with a fine gap, and must be narrow and made of small
  glyphs.

Held-out result: P = R = 1.000 (seeds 10–15), orientation ≥ 97 %.

**D-013 — Bubble segmentation.**
- Floating-range flood fill (±5) from seeds around the text, with every block's glyphs
  painted out first.
- Morphological opening (r ≈ ¼ glyph) severs tails.
- The ROI retries at 2.5×, 4.5× and 8× for large bubbles.
- Leak tests: ROI/page border, area > 20 × the text of all blocks inside, > 35 % of the
  page, solidity < 0.75. A leaked bright uniform bubble falls back to an ellipse
  (LEAK_FALLBACK); anything else becomes free_text.
- Narration = rectangularity ≥ 0.97 plus filled corners, measured before opening.
- Ground-truth bubble masks are the fillable interior (outline excluded), since that is
  what later stages use.

Held-out: mean IoU 0.991, min 0.95, types 161/161.

**D-014 — Reading order.** Panels come from recursive cuts along blank gutters in the page
image. Inside a panel, an XY-cut prefers horizontal cuts, and columns read right→left for
manga. `webtoon_ttb` = panels (full-width bands) top→bottom, then regions by y. Result:
28/28 pages exact.

**D-015 — OCR routing, as available here.** ja: manga_ocr (unavailable, W-001) →
EasyOCR `ja` with vertical reflow. ko: EasyOCR `ko`. zh: RapidOCR → EasyOCR `ch_sim`.
Engines recognise single lines. Vertical columns are reflowed: blank-row runs are merged
by gap-aware stroke attachment, so こ and う stay whole, and over-tall cells are split.
Held-out CER (seeds 10–12): ja-V 0.0 %, ja-H 0.0 %, ko 5.8 %, zh 0.0 %. The ko residue is
the EasyOCR recogniser confusing 었/없 and 줘/쥐 in Noto Sans KR; upscaling (×1.5–3) does
not change it.

**D-016 — Language detection.** Each language's first engine reads up to 4 of the largest
regions. Score = confidence × script consistency × (1 if the script vote agrees, else
0.5), plus a 0.15 vertical-text prior for ja. The pipeline aggregates votes per document
(folder/archive).

## Phase 3

**D-017 — Inpaint mask.**
- Start from the detector's glyph pixels.
- Bubbles and captions add "stray ink": pixels within 0.35 glyph of the text deviating
  > 30 levels from a median-filtered background. This catches anti-aliased stroke tips
  and dots. Visual inspection had found 1–6 px specks in 6/72 regions without it; after
  the fix, 0/72.
- Free text adds a contrasting halo (white outline over art), found by colour similarity
  to the pixels hugging the glyphs.
- Adaptive dilation clamp(½·stroke + 1, 2, 8), with stroke = 2 × P90 of the distance
  transform on a zero-padded mask. The padding fixes an overflow found by the tests.
- Clipped to the allowed zone: the bubble interior eroded 2 px, or dilated text plus halo
  margin for free text and leak fallbacks.

**D-018 — Background classification and strategy.** The ring 2–10 px around the mask,
inside the zone, is classified:
- uniform: robust per-channel spread (1.4826·MAD) ≤ 6 and outliers (> 20 levels) ≤ 3 %
  → solid fill (median colour; the 1-px feather blends only near-background pixels);
- gradient: robust residual of a per-channel planar fit ≤ 4 and residual outliers ≤ 3 %
  → Navier-Stokes (Telea fallback);
- otherwise texture → LaMa on a 96 px context crop (Telea fallback, INPAINT_FALLBACK).

Why the robust measures:
- Luminance std missed hue-shifting gradients (a bubble going cream → light blue).
- Plain std misfired on stray pixels.
- MAD alone called sparse screentone "uniform". Outlier fractions separate the cases
  cleanly: bubbles ≤ 0.6 %, textures ≥ 10.8 %.

Evidence (`spikes/out` and `.cache/out/texture_methods.png`): on screentone, LaMa
continues the dot pattern, while Telea and NS leave grey blobs. On smooth gradients, NS
is best (SP-E).

**D-019 — Text over dense hatching without a halo.**
- *Attempt 1*: a second detection pass on morphologically opened ink.
- *Attempt 2*: merging textured glyph groups into adjacent text.
- *Attempt 3*: a hybrid with the PP-OCR DB detector. On texture pages it reaches
  P = R = 1.0, but on regular pages P drops to 0.965 (from 0.986).
- Decision (rabbit-hole rule after three attempts): `auto` stays classical, and
  `detect.detector: hybrid` is offered for art-heavy pages.
- Known limitation: the classical detector can miss individual glyphs drawn straight onto
  1-px hatching without a halo. They then survive inpainting (fixable via GUI edit, the
  hybrid detector, or ctd).

**D-020 — Grayscale preservation.** When a crop is grey (R=G=B), the inpainted result is
averaged back to grey, so LaMa cannot introduce colour into B/W pages. A test covers
auto, Telea and LaMa.

## Phase 4

**D-021 — Provider protocol and orchestration.** A provider performs one request.
Batching, failover, caching and validation live in `TranslationService`. The default
chain is tm → google → mymemory → libretranslate (only with a URL) → local. Offline mode
drops network providers. Local MT implements `translate_many` (a true model batch, no
markers).

**D-022 — Resilience defaults.**
- Token bucket: 1 req/s, burst 2.
- Full-jitter backoff: base 1 s, ×2, cap 60 s, ≤ 6 attempts; `Retry-After` is a lower
  bound.
- Breaker: opens after 5 consecutive failures, 120 s cool-down, single half-open probe.
- Deadline: 25 s per call.

`call_with_deadline` uses a **daemon thread**. A `ThreadPoolExecutor` worker is joined at
interpreter exit, so an abandoned hung request would keep the process alive; that bug
was found and fixed while writing the deadline tests. deep-translator raises
`TooManyRequests` on HTTP 429 but does not expose `Retry-After`, so its 429s rely on
jittered backoff.

**D-023 — Page batching.**
- Format: `[1] … [2] …` on one line, because the Google web endpoint collapses newlines.
- Parsing tolerates full-width brackets and Arabic-Indic digits in the markers.
- Any missing, duplicated or out-of-order marker, or text before `[1]`, triggers the
  per-region fallback. A segment that fails validation is retried alone once.
- Chunk limits: google 4500, mymemory 450, libretranslate 4500, local 1500 characters.

**D-024 — Validation and flags.** An output is rejected when it is empty, identical to
the source, has an Arabic-letter ratio < 0.6, still contains CJK/Hangul, is longer than
8 × the source + 20, or has lost a glossary placeholder. When all providers reject a
region, it is flagged UNTRANSLATED and keeps its original pixels. OCR_SUSPECT regions are
not translated by default (`translate.translate_suspect: false`); they stay flagged for
review in the GUI.

**D-025 — Glossary.** Terms are replaced by `ZQX<n>X` tokens: Latin, unlikely to be
translated, and case-insensitive on restore. If the tokens are lost on every provider,
the text is re-translated without protection and flagged GLOSSARY_DEGRADED.

**D-026 — Cache.** SQLite (WAL) at `<cache>/translations.sqlite3`, plus an in-memory
layer. The key is sha256(cache version incl. MangaAR version | provider | src | tgt |
NFKC text). Identical sources on a page are translated once (dedupe).

**D-027 — normalize_ar.**
- NFKC, which also un-shapes presentation forms, so double shaping is impossible.
- Strips bidi controls (LRM, RLM, LRE, RLE, PDF, LRO, RLO, LRI, RLI, FSI, PDI, ALM),
  zero-width characters and tatweel.
- Letter mapping: keheh/gaf → kaf; Farsi yeh → yeh; heh goal/doachashmee → heh.
- Punctuation: `,` → `،` except decimal commas; `;` → `؛`; `?` → `؟`; 「」『』“” → «»;
  ellipses → `…`.
- Spacing is fixed; digit policy is western (default) or arabic_indic; decorations are
  re-appended; length is capped at 400 characters.

**D-028 — Bidi shim (A3).** python-bidi 0.6 (Rust) resolves embedding levels correctly,
including bracket pairs (N0), but does not apply L4 mirroring, so `على (OK)` would render
as `)OK(`. The shim reads the resolved levels from `get_display_inner(text, "R", debug=True)`
(per UTF-8 byte → per character), then applies L2 reordering and L4 mirroring (the
`bidi.mirror.MIRRORED` table) itself. Lines without a mirrorable character use the plain
Rust `get_display` fast path. Older pure-Python releases mirror on their own and serve as
the fallback. python-bidi is LGPL-3.0; it is used as an unmodified dependency, which is
compatible with the MIT licence of this project.

**D-029 — Reshaper configuration (A4).** An explicit configuration is used:
- the four lam-alef ligatures are on; ALLAH and the word/sentence ligatures are off, so
  joining stays predictable;
- tatweel is deleted and harakat are stripped (configurable);
- `use_unshaped_instead_of_isolated` is chosen per font from its cmap. A font without
  isolated presentation forms gets base letters, which render as isolated glyphs.

Fonts that lack contextual presentation forms entirely (BalooBhaijaan2) are marked
not `basic_ok` and are only offered on the RAQM path.

**D-030 — ARVS thresholds (L5/L6).**
- L5 counts connected components of ink < 200 (8-connectivity). A lower threshold split
  Cairo's hairline anti-aliased joins into extra components (محمد counted 3 instead of 1).
- The L5 direction oracle uses the probe "اب", whose first letter's ink must lie on the
  right. An earlier "لا" probe failed for Changa's ligature design.
- L6 compares per-word ink between the BASIC path and a HarfBuzz (RAQM) oracle. The
  default threshold is IoU ≥ 0.90; Amiri needs 0.78 because its HarfBuzz rendering uses
  contextual alternates the presentation-form path cannot reproduce (legible, same
  letters, but different glyph variants).
- L6 negative controls (unshaped text) must score ≤ 0.70 and at least 0.25 below the
  correct render.

**D-031 — ARVS L7 (OCR round-trip).** Setup:
- EasyOCR `['ar','en']` (weights from GitHub releases; the only Arabic OCR reachable here);
- 21 phrases at 48/60/72 px, plus 4 phrases laid out inside an elliptical bubble;
- every `basic_ok` bundled font;
- score = Levenshtein similarity after `normalize_ar` with whitespace removed.

Thresholds:
- ≥ 90 % of all samples must reach similarity ≥ 0.8, and ≥ 75 % per font. A single font
  may have a few misreads; EasyOCR confuses some display-font glyphs.
- Unshaped and not-reordered negative controls must have a mean similarity < 0.4.

Measured:

| Font | Samples ≥ 0.8 | Mean similarity |
|---|---|---|
| NotoNaskhArabic | 25/25 | 1.00 |
| NotoSansArabic | 25/25 | 0.99 |
| Amiri | 23/25 | 0.96 |
| Cairo | 25/25 | 0.98 |
| Tajawal | 24/25 | 0.97 |
| TajawalBold | 25/25 | 0.99 |
| Almarai | 25/25 | 1.00 |
| AlmaraiBold | 25/25 | 1.00 |
| Changa | 24/25 | 0.97 |
| Lalezar | 25/25 | 0.99 |

The negative controls scored 0.14–0.19 at calibration.

**D-032 — Fit and overflow ladder (A6–A9).**
- Strategy "shape": each line's available span is the padded bubble mask's longest run
  over that line's ink band, so text follows the outline. Strategy "rect": the largest
  inscribed rectangle, used for LEAK_FALLBACK or when requested.
- Word widths are measured once at 100 px and scaled. Arabic joining never crosses a
  space, so a line width is the sum of its words plus spaces. The best candidate is then
  re-measured exactly, and candidates are verified in cost order.
- Font size: integer binary search, then a probe of up to 3 sizes above the result.
  Feasibility is not strictly monotonic with discrete line counts; without the probe,
  "shape" occasionally lost to "rect".
- Ladder order: line-spacing-floor → condensed (only when the font has a `wdth` axis)
  → padding-floor → extend-uniform → hard-floor.
  - extend-uniform: bubbles with a trusted mask use their unpadded interior and never
    leave it; free text flood-fills uniform background on the clean page.
  - hard-floor: minimum size, greedy wrap, flag OVERFLOW_RISK. The block is shifted to
    stay inside the image and is never clipped.
- Colours: black or white by WCAG contrast (≥ 4.5 : 1). Free text always gets an
  outline; a contrasting outline is also added when the target contrast cannot be met.
- Performance: a typical region lays out in ≤ ~45 ms cold (the A13 target is < 50 ms),
  thanks to the `shape_line` LRU cache and the word-metric cache keyed by a stable
  identity.

**D-033 — Morphology borders.** `cv2.erode` treats pixels outside the array as
foreground by default. Tight bubble masks touch their crop edges at the four extremes,
so padding erosion kept full-width spikes reaching the outline there (found on the L8
specimen sheet). `geometry.erode()` uses a constant 0 border; it is now used for layout
padding, the inpainting allowed zone and inscribed safe boxes. Regression tests cover
both the layout and the inpainting cases.

**D-034 — Page composition.** The clean image stores every detected text inpainted; this
includes untranslated regions. At composition time, regions without Arabic text get their
original pixels back inside their inpaint mask:
- this covers untranslated, skipped and TYPESET_FAILED regions;
- `--erase-untranslated` keeps them erased instead;
- a GUI edit or `rerender` can therefore typeset a previously untranslated region without
  re-inpainting;
- typesetting errors are isolated per region (flag TYPESET_FAILED; the page continues).

**D-035 — SFX handling.** Every stage skips `sfx` regions, which is the default
(`detect.sfx: skip`) and leaves them untouched. With `--sfx translate` the pipeline
re-types SFX regions as `free_text` right after segmentation and flags them `FROM_SFX`,
so provenance survives in the sidecar. They then get free-text treatment: OCR,
inpainting, and typesetting with an outline over artwork.

**D-036 — Output layout.**

| Output | Path |
|---|---|
| page image | `<out>/<relative dir>/<stem>_ar.<ext>` |
| sidecar | `<stem>_ar.mangaar.json`, next to the page |
| work images | `.mangaar/<stem>.source.png` and `.mangaar/<stem>.clean.png` |
| debug artifacts | `<out>/debug/<relative dir>/<stem>/` |
| reports | `report.json` + `report.md` at the output root |

- Relative directories are preserved: input folder name / sub-folders, and
  `<archive stem>/<member dirs>` for archives.
- The work images are the decoded original and the page with all text removed. They
  make a sidecar self-contained for `rerender` and the GUI; archive members and GUI
  uploads have no stable original file. The cost is two lossless PNGs per page.
- The sidecar also stores a snapshot of the configuration (`settings`). `rerender` uses
  it plus the explicit flags, so a page re-renders with the settings it was produced
  with.
- With `--format cbz`, pages are grouped into one book per input archive, input folder,
  or parent folder of loose files, written as `<group>_ar.cbz` with PNG members named
  `<member dir>/<stem>_ar.png`. The sidecars live in `<group>/`.
- The output directory is excluded when scanning inputs, so a rerun into an output
  folder inside the input folder does not translate its own results.

**D-037 — Page status and exit codes.**
- Successes: `ok`, `degraded` (exported, but some region is UNTRANSLATED,
  TYPESET_FAILED or OCR_FAILED, or failed inpainting), `no_text` (exported unchanged,
  E21) and `resumed`.
- Failures: `skipped` (unreadable or unsupported input), `failed` (crashed) and
  `cancelled`.
- Exit code: 0 when every page succeeded, 2 when some did not or the run was cancelled,
  1 when none succeeded (or on a fatal error). Degraded pages keep exit code 0; the
  report counts their flags and notes how to fix them, e.g. installing the local MT model
  for E4.

**D-038 — Resume (E19).** A page is "done" when its sidecar has the same config hash
(runtime-only keys excluded; the MangaAR version included) and the same source SHA-256.
- Done page whose output exists: skipped.
- Output missing, or a CBZ member: re-rendered from the work images, without detection.
- Degraded page: only its UNTRANSLATED regions are sent to translation again, then the
  page is re-rendered. A rerun after connectivity returns therefore fills the gaps
  cheaply.
- `--force` reprocesses everything.

**D-039 — Source-language vote (E6).** An explicit `--source` wins. Otherwise
`detect_language` scores (confidence × script agreement, plus a vertical-text prior for
Japanese) accumulate per book (archive or folder) until three pages have voted; later
pages reuse the leading language. A per-region `override.source_lang` wins for that
region. OCR and translation are grouped by region language.

**D-040 — Isolation and cancellation.**
- Region level: an OCR crash flags OCR_FAILED. An inpainting failure flags SKIPPED;
  composition then restores that region's original pixels. A translation exception
  flags UNTRANSLATED. A typesetting error flags TYPESET_FAILED.
- Page level: any other exception fails only that page.
- Cancellation (`CancelToken`) is cooperative: it is checked between pages, between
  regions (OCR and inpainting) and before typesetting. The run still writes the CBZs and
  reports for what was finished.

**D-041 — GUI.**
- One `GuiController` per launch. Each run gets a directory in a temporary workspace
  that is deleted at exit (atexit), and Gradio's copied files use
  `delete_cache=(3600, 3600)`.
- All events share one queue worker (`default_concurrency_limit=1`); only Cancel
  bypasses the queue. Stages are cached for the current configuration only (one model
  set in memory).
- Telemetry: `GRADIO_ANALYTICS_ENABLED=False` is set before Gradio is imported, and
  `analytics_enabled=False` is also set on the Blocks.
- Network exposure: the GUI binds to `127.0.0.1` with `share=False`; any other host logs
  a warning.
- `allowed_paths` covers only the workspace.
- Review & Edit writes the table edits into region overrides (text, font, size, skip),
  then calls the same `rerender` as the CLI.
- Downloads are a ZIP of the outputs (without work dirs or debug files).

**D-042 — File permissions.** `mkstemp` creates 0600 files; atomic writes now chmod the
temp file to `0666 & ~umask` before the rename, so outputs get normal permissions.

## Waivers

**W-001 (SP-B / P2 OCR JA via manga-ocr).** Reason: huggingface.co blocked. Risk: vertical
Japanese OCR quality on real pages is lower without manga-ocr. Mitigation: EasyOCR `ja` +
vertical reflow. The adapter is exercised by `pytest -m integration
tests/integration/test_ocr_engines.py -k manga` on a host with Hub access.

**W-002 (SP-C PaddleOCR).** Reason: every Paddle model host is blocked, and pipeline
creation needs extras. Mitigation: RapidOCR (zh) and EasyOCR (ko/ja/zh fallback). The
Paddle adapter stays optional (`[paddle]` extra) and is marked unverified.

**W-003 (P4 live network smoke).** Reason: translate.google.com and
api.mymemory.translated.net are blocked. Mitigation: full resilience suite with fakes.
The live smoke `pytest -m network tests/integration/test_translate_live.py` has to be run
on an unrestricted host.

**W-004 (P4/P6 offline local-MT smoke, DoD #5; Gate P6 "offline E2E through the local MT
model").** Reason: the Marian/M2M100 weights are
only reachable on huggingface.co. Mitigation: the local provider is implemented and unit
tested with a fake model. Offline E2E is proven with the TM provider. On an unrestricted
host, `manga-arabic models download local-mt` followed by `pytest -m integration -k
local_mt` completes the gate.

Gate P6 evidence under W-004: `tests/e2e/test_cli_real.py::
test_offline_without_local_model_keeps_originals_and_explains` runs `translate --offline
--providers local`. Without the weights it verifies E4 (regions UNTRANSLATED, originals
kept, exit 0, and the report explains `models download local-mt`). With the weights
installed, the same test asserts that every region is translated offline. The full
offline pipeline is also proven with the TM provider: `demo`, and `--offline` in the CLI
E2E tests.
