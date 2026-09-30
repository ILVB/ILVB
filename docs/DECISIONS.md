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
(~4.9 MB) sit in `assets/fonts/` with OFL texts in `assets/fonts/licenses/` and a
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

**W-004 (P4/P6 offline local-MT smoke, DoD #5).** Reason: the Marian/M2M100 weights are
only reachable on huggingface.co. Mitigation: the local provider is implemented and unit
tested with a fake model. Offline E2E is proven with the TM provider. On an unrestricted
host, `manga-arabic models download local-mt` followed by `pytest -m integration -k
local_mt` completes the gate.
