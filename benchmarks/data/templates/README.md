# Real gold data: what to provide and how

Put the filled files and the page images in `benchmarks/data/gold_real/`. That directory is
**gitignored**; nothing in it is ever committed or embedded in committed reports (PD-8).
Only provide pages you have the right to use for private evaluation.

| File (copy the template, drop the `.template`) | Minimum | Purpose |
|---|---|---|
| `pages.csv` + the page images (`images/<file_name>`) | 30 pages, ≥ 150 text regions in total, ≥ 2 series preferred | real-page OCR/detection/inpainting/typesetting evaluation |
| `regions.jsonl` (one JSON object per line; see `regions.schema.json`) | every text region on those pages | detection boxes/polygons, region type, exact transcription, reading order, speaker |
| `arabic_references.csv` | ≥ 100 regions | human-authored Modern Standard Arabic reference translations (G-TR-1) |
| `glossary.yaml` | per series: names, recurring terms, places, honorific policy | glossary/name adherence (G-TR-2), Series Bible seed |
| `speakers.csv` | per series: every speaking character | gender/number agreement (G-TR-3) |

Rules:
- **Transcriptions** are exactly what is printed (keep punctuation, `…`, `！？`; one region = one
  bubble/caption/SFX; join its lines with a single space for horizontal text, no separator for
  vertical Japanese). Unreadable → leave `text` empty and set `"illegible": true`.
- **Region polygons**: pixel coordinates in the original image, clockwise, at least 4 points.
  A tight box around the text is fine for `text_polygon`; `bubble_polygon` is optional but
  valuable (the bubble's inner outline).
- **Arabic references** must be written by a person (not machine translation, not post-edited
  MT). If a translator used MT as a draft, mark `post_edited_mt=yes` so it can be excluded.
- **Splits** are made by the harness (by series/page, never splitting one page); do not
  pre-split. The test split is sealed on arrival (hash-recorded, never inspected).

Annotation tools: any polygon tool works (e.g. CVAT, LabelMe). LabelMe JSON or CVAT XML
exports are also accepted; a converter will map them to `regions.jsonl`.
