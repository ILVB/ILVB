# Metric definitions (v0.2.0 harness and Gate 1)

Each metric is computed by code in `benchmarks/metrics/` from raw result files
(`benchmarks/results/`). These definitions are fixed before any Phase 1 implementation
(PD-3). Changing one needs an ADR and a re-run of both baseline and candidate.

## OCR and detection (`text.py`, `detection.py`)
| Metric | Definition |
|---|---|
| CER | Micro-averaged character edit distance (jiwer) ÷ reference length, after NFKC and removal of all whitespace. Aggregated as Σedits / Σlength (never a mean of per-region rates). SFX regions are excluded from CER and reported separately. |
| WER | Same for whitespace tokens (English and Korean only). |
| Detection P/R/F1 | Axis-aligned boxes; greedy one-to-one matching by descending IoU; a match needs IoU ≥ 0.5. |
| G-OCR-1 categories | standard bubbles = flat_white + tails_overlap; vertical text = vertical_ja; low-contrast = low_contrast; screentone = screentone; stylized fonts = stylized (synthetic_v1 categories). dark_bubble, text_on_art, sfx and tiny_text are reported per category but not constrained by G-OCR-1. Fixed at step 1.0.2, before any candidate OCR change. |
| Reading-order accuracy | Over IoU-matched regions only: GT and predicted orders are converted to ranks among the matched set; accuracy = regions whose predicted rank equals their GT rank ÷ matched regions. |

## Inpainting (`inpaint.py`)
| Metric | Definition |
|---|---|
| G-INP-1 background integrity | Pixels (any channel) differing from the input outside `dilate(declared erase mask, 2 px)`. It must be 0. Mask overreach (declared mask outside the GT text mask dilated by 6 px) is reported as a diagnostic. |
| G-INP-2 SSIM / PSNR | Computed on the GT text bbox + 8 px margin, erased vs GT clean page (scikit-image; window 7 or the largest odd size that fits; PSNR capped at 100 dB). SFX is excluded. LPIPS is NOT-MEASURABLE until backbone weights are supplied (ADR-0001 #8). |
| G-INP-3 residual text | A region counts as residual if the RapidOCR PP-OCR DB detector finds text in its crop (probe) **or** the pixel check fires: more than max(8, 2 % of text-mask area) pixels inside the text mask dilated by 2 px deviate from the GT clean page by more than 48 levels. The OR stops a candidate that adopts the same DB detector from grading itself. |

## Typesetting (`typeset.py`; run in `typeset_gt` mode: GT geometry, identical Arabic text)
| Metric | Definition |
|---|---|
| G-TYPE-1 ink outside safe | Rendered ink pixels outside the GT safe mask (bubble eroded by max(3, 6 %) or a dilated hull for art text). A region passes when this is 0. |
| G-TYPE-2 legibility | Font size ≥ 1.1 % of page height (15.4 px on 1400 px pages). |
| Raggedness | For every line except the last: slack = (available − line width) ÷ available, where available = the minimum, over the line's pixel rows, of the longest safe-mask run in that row. Report the mean of slack². Lower is better. |
| Orphans | Multi-line blocks whose last line is a single word. |
| G-TYPE-4 hyphenation | Arabic lines never end with a hyphen (U+002D, U+2010, U+2011, U+00AD). |
| Overflow | Regions the typesetter flagged as overflow/needs-review. |

## Translation (`translation.py`)
| Metric | Definition |
|---|---|
| BLEU | sacreBLEU corpus BLEU, `tokenize="intl"`, default smoothing. Multiple references per segment; uneven counts are padded by repeating the segment's last reference. |
| chrF++ | sacreBLEU `CHRF(word_order=2)`. |
| METEOR (adapted) | NLTK METEOR (α=0.9, β=3, γ=0.5) on `\w+` tokens, matching exact forms and ISRI Arabic stems. The WordNet synonym stage is disabled (no Arabic WordNet), so scores are not comparable with English METEOR. |
| Significance | sacreBLEU paired bootstrap resampling (`test_type="bs"`, 1000 samples, sacrebleu's fixed seed 12345 via `SACREBLEU_SEED`) for BLEU and chrF++, candidate vs baseline. |
| Reference kind | Every report prints `SILVER-REFERENCE, comparative only` for synthetic references (ADR-0002). GOLD supersedes it when delivered. |

## Statistics and performance (`stats.py`)
- Confidence intervals: 95 % percentile bootstrap over pages, 1000 resamples, seed 20260930.
- Latency: p50, p95 and max of per-page wall time per stage. Peak RSS is sampled every 5 ms from `/proc/self/statm`.
