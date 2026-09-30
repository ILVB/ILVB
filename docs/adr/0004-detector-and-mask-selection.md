# ADR-0004: Text detector for the v2 profile (step 1.0.3)

Status: accepted. `detect.detector: auto` resolves to `db_primary` under
`engine.profile: v2`. Under `legacy` it stays `classical` (v0.1.0, D-011).

## Evidence (synthetic_v1; SILVER data, comparative only)
Harness: `run_benchmark --version candidate --profile legacy-offline --modes erase --set
detect.detector=…`. CER is micro-averaged, detection is matched at IoU 0.5, and mask IoU is
the mean over matched regions. Timings in rows marked * are invalid: those runs overlapped
other runs on the 4-core host.

Dev (40 pages):

| detector | CER | P | R | F1 | mask IoU | SSIM | residual | erase page p50/p95 s |
|---|---|---|---|---|---|---|---|---|
| classical (v0.1.0) | 0.250 | 0.780 | 0.896 | 0.834 | 0.717 | 0.827 | 0.361 | 0.60/4.14 |
| hybrid | 0.235 | 0.793 | 0.956 | 0.867 | 0.709 | 0.841 | 0.348 | 0.91/4.66 |
| rapid (PP-OCR DB) | 0.223 | 0.980 | 0.980 | 0.980 | 0.712 | 0.852 | 0.330 | 0.82/3.47 |
| ctd (GPL-3.0, opt-in) | 0.217 | 0.982 | 0.880 | 0.928 | 0.669 | 0.837 | 0.356 | 34.3/95.0 * |
| db_primary, first rule | 0.224 | 0.976 | 0.976 | 0.976 | 0.718 | 0.853 | 0.313 | 24.3/83.9 * |
| **db_primary (adopted)** | 0.228 | 0.984 | 0.984 | 0.984 | 0.711 | 0.853 | 0.326 | 0.96/3.87 |
| db_primary, vertical-only swap | 0.229 | 0.984 | 0.984 | 0.984 | 0.718 | 0.855 | 0.313 | 0.95/3.79 |

G-OCR-1 categories, CER on dev (classical → rapid → adopted db_primary):
standard bubbles 0.089 → 0.068 → 0.084; vertical 0.042 → **0.141** → 0.042;
low-contrast 0.151 → 0.118 → 0.151; screentone 0.364 → 0.359 → 0.364;
stylized 0.788 → 0.707 → 0.707. The first db_primary rule regressed screentone
(0.419) and vertical text (0.058). The vertical-only variant regressed screentone
(0.401: rapid geometry produced garbage Korean OCR on a screentone page). Both were
rejected.

Val (40 pages), frozen v0.1.0 baseline vs adopted db_primary:

| | CER | F1 | order | SSIM | residual |
|---|---|---|---|---|---|
| v0.1.0 | 0.243 | 0.850 | 0.817 | 0.835 | 0.313 |
| db_primary | **0.192** | **0.992** | 0.831 | 0.863 | 0.295 |

No category is worse on val (Δ CER in points: screentone −17.9, text_on_art −27.9,
tiny_text −3.8, stylized −2.9, all others ±0.0). The detect stage runs 0.68 s p50 and
1.20 s p95 per page, within the budgets of 1.29 s and 1.76 s.

## Decision
`db_primary` = PP-OCR DB decides which blocks exist (it rejects texture false
positives). Classical refines those blocks only on clean backgrounds (hybrid's texture test):
a classical block with IoU ≥ 0.7 supplies the geometry, and a larger classical block
(≥ 1.1×, ≤ 3×) covering ≥ 60 % of DB blocks extends them. A classical block never
enters on its own. The thresholds were set on dev and confirmed on val.

- `ctd` stays opt-in: GPL-3.0 weights, lower recall (0.880), and worse low-contrast and SFX-page
  results.
- `rapid` alone regresses vertical text by 9.9 points, which G-OCR-1 forbids.
- Pixel masks: every detector builds its glyph masks from the same polarity-aware
  classical ink, so mask IoU hardly moves (0.67–0.72). The DB probability map is a
  region-level (shrunk text-area) map and cannot supply glyph-level masks. Mask quality is
  handled at 1.3.1 (stroke-adaptive dilation, polarity) and judged by the inpainting
  metrics.

## Consequences
- Detection F1 no longer limits G-OCR. The remaining CER comes from recognition
  (stylized 0.72, text_on_art 0.36, tiny_text 0.28 on val). Steps 1.0.4 and 1.0.5 target it.
- tiny_text on dev got worse (0.158 → 0.205; DB misses some tiny lines). The category is not
  constrained by G-OCR-1, and it improved on val. It is watched at 1.0.4 (upscaling).

## Update — recognizer selection per script (step 1.0.5)
v0.1.0 has no English route. English pages went through its `auto` path (a ja/zh/ko engine
plus Latin pass-through), and recognised lines were joined without spaces except in Korean.
v0.2.0 adds `en` as an explicit source with `ocr.engines.en: [rapid, easyocr]` (the bundled
PP-OCR recogniser reads Latin; EasyOCR `english_g2`, Apache-2.0, fetched from GitHub
releases, MD5-checked by easyocr). English lines are joined with spaces, and Latin text is
not passed through when English is the source. Harness: the candidate adapter passes `en`
explicitly only under the `v2` profile, so `legacy` stays byte-identical.

| dev, v2 profile (db_primary) | CER | WER | en CER / WER |
|---|---|---|---|
| no English route | 0.228 | 1.003 | 0.261 / 1.00 |
| en: rapid → easyocr (adopted) | 0.182 | 0.491 | 0.149 / 0.20 |
| en: easyocr → rapid | 0.209 | 0.648 | 0.215 / 0.44 |

Val, frozen v0.1.0 → v2 profile: CER 0.243 → **0.129** (−47 % relative; G-OCR-1 needs ≥
15 %), WER 0.821 → 0.292, detection F1 0.850 → 0.992. No category is worse: vertical text
0.047 → 0.047, screentone −21.9 points, stylized −25.5, text_on_art −40.1, low-contrast −5.3,
standard bubbles −4.1. The remaining weak spots are stylized (0.50), tiny_text (0.26) and
text_on_art (0.23). Korean CER is 0.21.

## Update — orientation, DB line geometry and texture suppression (steps 1.0.3–1.0.4)
Three v2 upgrade flags (all `auto`: on under `v2`, off under `legacy`):
- `detect.db_geometry`: when a block's DB line boxes are clearly wide (≥ 1.5 ×) or tall, they
  fix its orientation and become its lines. Before this, 11 of 229 horizontal dev regions
  were judged vertical and read as scrambled columns. Rows of screentone dots were also
  segmented into spurious "lines" (e.g. "벼무 무 하 ; ; ; @ @ < < < …" joined before the text).
  db_primary never lets a classical block flip a DB block's orientation.
- `ocr.clean_texture`: on OCR crops only, when small-mark density is ≥ 15 per 1000 px
  (screentone median 19.5, flat bubbles ≈ 8.5 on dev crops), glyph strokes and adjacent
  punctuation are kept on a flat background. The page itself is never modified.

| dev, v2 profile | CER | WER | screentone | ko | en |
|---|---|---|---|---|---|
| English route | 0.182 | 0.491 | 0.456 | 0.416 | 0.149 |
| + orientation from DB boxes | 0.159 | 0.509 | 0.512 | 0.395 | 0.126 |
| + texture suppression | 0.158 | 0.502 | 0.493 | 0.389 | 0.125 |
| + DB boxes as lines (adopted) | **0.118** | 0.297 | **0.083** | **0.213** | 0.095 |
| same, texture suppression off | 0.120 | 0.300 | 0.097 | 0.216 | 0.096 |

Val, frozen v0.1.0 → adopted v2: CER 0.243 → **0.118** (−51 %), WER 0.821 → 0.274,
detection F1 0.850 → 0.992, SSIM 0.835 → 0.953, residual 0.313 → 0.141. Every category is at
or below v0.1.0 (vertical 0.047 = 0.047, screentone −27.6 points, stylized −28.4, text_on_art
−42.4). Relative to the English-route step, English (0.103 → 0.108) and tiny_text
(0.256 → 0.269) moved slightly; both remain well below v0.1.0.
