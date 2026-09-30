# MangaAR User Guide

This guide covers everyday workflows. For installation see the [README](../README.md);
for problems see [TROUBLESHOOTING](TROUBLESHOOTING.md).

## 1. The pipeline in one minute

For every page, MangaAR:

1. **Loads** it.
   - Formats: PNG, JPEG, WebP, BMP and TIFF, or a CBZ/ZIP member.
   - Normalisation: EXIF rotation is applied, transparency is flattened onto white, and
     16-bit or palette images are converted.
   - Webtoon strips (taller than 2.5× their width, or taller than 4096 px) are processed
     in overlapping tiles.
2. **Detects** text blocks.
   - Region types: `bubble`, `narration`, `free_text` or `sfx`.
   - It segments each bubble's interior and orders the regions in reading order:
     `manga_rtl` for Japanese, `comic_ltr` for Chinese, `webtoon_ttb` for Korean.
3. **Reads** the text with OCR.
   - Engines: EasyOCR or manga-ocr for Japanese, EasyOCR for Korean, PP-OCR (RapidOCR)
     for Chinese.
   - Vertical columns are reflowed for engines that only read horizontally.
   - Suspicious results (empty, wrong script, repetition loops) are retried with the next
     engine, and flagged if nothing better is found.
4. **Removes** the text.
   - Uniform bubbles: filled with their exact colour.
   - Textured backgrounds: LaMa, when installed.
   - Otherwise: OpenCV inpainting.
   - Pixels outside the text mask never change, and bubble outlines are protected.
5. **Translates** to Arabic through the provider chain (see §5).
6. **Letters** the Arabic.
   - The text is wrapped on the logical string.
   - Each line is shaped, with joined letters and lam-alef ligatures.
   - Lines are ordered right-to-left, with numbers and Latin kept left-to-right and
     brackets mirrored.
   - The largest font size that fits the bubble's shape is chosen, and the lines follow
     round bubbles.
   - Text colour is black or white, whichever contrasts best. Free text over artwork gets
     an outline.
7. **Exports** the page, its sidecar JSON, two work images and the batch report.

## 2. Translating pages

```bash
manga-arabic translate chapter01/ -o out/
```

- **Inputs:** any mix of image files, folders (searched recursively; dot-folders are
  ignored) and `.cbz`/`.zip` archives.
- **Order:** pages run in natural order (page2 before page10).
- **Relative paths:** `chapter01/sub/p1.png` becomes `out/chapter01/sub/p1_ar.png`.
  Archive members go under `out/<archive name>/…`.
- **Output format:** `--format png` (default, lossless), `jpg` (quality 95), `webp`, or
  `cbz`.
  - With `cbz`, each archive or input folder becomes `out/<name>_ar.cbz`, and its
    sidecars go to `out/<name>/`.
  - ICC colour profiles are preserved, and grayscale pages stay grayscale.
- **Source language:** `--source auto` (default) votes over the first three pages of each
  book. Pass `--source ja|ko|zh` when you know it, which is faster and never wrong.
- **Reading order:** `--reading-order` overrides the per-language default. It only
  affects translation context (neighbouring bubbles), never where text is placed.
- **Sound effects:** by default (`--sfx skip`) large stylised SFX are left untouched.
  `--sfx translate` treats them as free text: erased, translated and lettered with an
  outline.

### Reports and exit codes

`out/report.md` lists every page with its status, detected language, number of regions
translated, flags and time. `out/report.json` has the same data for scripts.

| Status | Meaning |
|---|---|
| `ok` | everything translated and lettered |
| `degraded` | exported, but some regions are untranslated, or OCR/inpainting/typesetting failed there; see the flags |
| `no_text` | no text found; the page is exported unchanged |
| `resumed` | already done with the same settings (`--resume`) |
| `skipped` | not an image, or unreadable (corrupt, truncated, too large) |
| `failed` | an unexpected error on that page; the batch continued |

Exit code `0` means every page succeeded (including `degraded`), `2` a partial success,
and `1` that nothing was processed.

### Resuming and re-running

`--resume` skips pages whose sidecar was produced with the same effective settings and
the same source image.
- Pages whose output was deleted are re-rendered from their work images in a fraction of
  a second.
- `degraded` pages only retry translating their untranslated regions. This is useful
  after a rate limit or a network outage.
- Changing any setting that affects the result (font, preset, providers…) reprocesses
  the page.
- `--force` always reprocesses.

## 3. Reviewing and fixing translations

### In the GUI
1. `manga-arabic gui`, then translate your pages on the **Translate** tab.
2. Open **Review & Edit** and pick a page. Each row is one region, in reading order:
   `id`, `type`, `source text`, **`Arabic text`**, **`font`**, **`size`**, **`skip`** and
   `flags`.
3. Edit the bold columns:
   - Arabic text: type your own; clearing it goes back to the machine translation.
   - Font: any key from `manga-arabic fonts list`.
   - Size: pixels; leave it empty for automatic fitting.
   - Skip: tick it to leave the original text.
4. Press **Re-render page**. Only the lettering runs again, so this takes well under a
   second.
5. **Re-translate region** (with a region id) asks the providers again, for example after
   fixing the glossary.

Rows flagged `OCR_SUSPECT` were not translated automatically; check their source text.
`OVERFLOW_RISK` means the text needed the smallest allowed size; shorten it or pick a
narrower font.

### From the command line
Each page's `<page>_ar.mangaar.json` contains a `regions` list. Every region has an
`override` object:

```json
"override": {"text": "نص جديد", "font": "Cairo", "size_px": 28, "source_lang": null, "skip": false}
```

Edit it, then run:

```bash
manga-arabic rerender out/chapter01/p1_ar.mangaar.json            # apply the edits
manga-arabic rerender out/chapter01/*_ar.mangaar.json --font Tajawal   # new font everywhere
```

`rerender` reuses the settings the page was produced with (stored in the sidecar) plus
the flags you pass (`--font`, `--digits`, `--erase-untranslated`). It never re-runs
detection, OCR, inpainting or translation.

## 4. Fonts and typography

All bundled fonts are SIL OFL:
- Naskh: Noto Naskh Arabic (default), Amiri.
- Sans: Noto Sans Arabic, Cairo, Tajawal (Regular/Bold), Almarai (Regular/Bold).
- Display: Changa, Lalezar, Baloo Bhaijaan 2.

Symbols such as ♡ ♥ ☆ ★ ♪ fall back to Noto Sans Symbols automatically.

- `manga-arabic fonts check` verifies each font's coverage. "RAQM-only" fonts (Baloo
  Bhaijaan 2) need libraqm, which Pillow wheels include on most platforms.
- `--digits arabic_indic` writes ٠١٢٣…; the default is Western digits.
- Harakat (short-vowel marks) are removed by default, because the default rendering path
  cannot position them.
- `typeset.render_path: raqm` in a config file switches to HarfBuzz shaping when libraqm
  is available.

The fitting strategy is `typeset.strategy: shape` (lines follow the bubble outline) or
`rect` (largest inscribed rectangle). When text does not fit, MangaAR tries these steps
in order:
1. tighter line spacing;
2. a condensed width, for fonts that have one;
3. smaller padding;
4. growing into surrounding uniform background, never across outlines;
5. the minimum size, flagged `OVERFLOW_RISK`.

Text is never clipped or drawn off the page.

## 5. Translation providers, glossary and translation memory

Choose providers with `--providers` or `translate.providers` in a config file. They are
tried in order:

| Provider | Needs network | Notes |
|---|---|---|
| `tm` | no | exact matches from your translation memory |
| `google` | yes | Google web endpoint via deep-translator; no key; rate-limited |
| `mymemory` | yes | MyMemory free tier (short daily quota) |
| `libretranslate` | yes | only when `translate.libretranslate_url` points to an instance (e.g. self-hosted) |
| `local` | no | Marian OPUS-MT (pivot via English) or M2M100 (`translate.local_model`) |

**Glossary** (`--glossary names.yaml`): terms are protected during translation and
replaced by your Arabic rendering. Longer terms win. JSON, YAML or 2-column CSV:

```yaml
ルフィ: لوفي
先輩: سينباي
さん: سان
```

**Translation memory** (`--tm memory.json`): whole-bubble translations, matched after
Unicode normalisation (whitespace ignored). Same formats. It is ideal for recurring
phrases or for fully offline work.

`MANGAAR_CACHE_DIR/translations.sqlite3` caches every machine translation. Delete it to
force fresh translations.

## 6. Webtoons, archives and large batches

- **Long strips** are tiled automatically. A bubble crossing a tile seam is processed
  once, from a tile that contains it completely. If memory runs out, the tiles shrink and
  the page is retried.
- **Archives**:
  - members are read in memory, in natural order;
  - non-images are ignored;
  - unsafe paths (`../`, absolute) reject the whole archive;
  - size limits guard against zip bombs (`input.archive_*` settings).
- **Big batches**: every page is independent, so a failure never stops the run. Use
  `--resume` to continue an interrupted batch.

## 7. Configuration

```bash
manga-arabic translate pages/ -o out/ --config my.yaml
```

`my.yaml` only needs the keys you change. Every key and its meaning is in
[`configs/default.yaml`](../configs/default.yaml), and unknown keys are rejected. Example:

```yaml
preset: quality
typeset:
  font: Cairo
  line_spacing: 1.2
translate:
  providers: [tm, local]
  glossary_file: names.yaml
```

Environment variables:

| Variable | Effect |
|---|---|
| `MANGAAR_CACHE_DIR` | where models and caches live |
| `MANGAAR_OFFLINE=1` | same as `--offline` |
| `MANGAAR_DEVICE=cpu` | same as `--device cpu` |
| `MANGAAR__typeset__font=Amiri` | any key, as `MANGAAR__<section>__<key>` |

`--debug` writes to `out/debug/<page>/`:
- detections, text masks and inpaint masks;
- an inpainting-difference heat-map and the clean page;
- layout boxes;
- the transparent Arabic text layer.

## 8. Offline and privacy

`--offline` (or `runtime.offline: true`) guarantees that nothing touches the network: no
model downloads and no online providers. Translation then comes from `tm`, from `local`
(after `manga-arabic models download local-mt`), or from the SQLite cache.

In online mode only the recognised text is sent to the translation providers you enable,
never images. Model downloads come from GitHub releases or the Hugging Face Hub. The
GUI binds to 127.0.0.1, and Gradio analytics are disabled.
