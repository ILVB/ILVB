"""Quality benchmark on held-out synthetic pages → docs/QUALITY_REPORT.md (T6).

Usage: python scripts/benchmark.py [--seeds 100 105] [--no-ocr] [--out docs/QUALITY_REPORT.md]

Sections: detection P/R @IoU0.5, bubble-mask IoU, region typing, reading order, OCR CER
per language/orientation, inpainting (residual rate, invariants, per-preset timings),
translation failover (fake providers), typesetting (overflow / fit / ARVS joining) and
per-stage timings with peak memory.
"""

from __future__ import annotations

import argparse
import os
import platform
import resource
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from manga_ar import __version__, synth
from manga_ar.config import load_config
from manga_ar.detect.bubble import FloodBubbleSegmenter
from manga_ar.detect.classical import ClassicalDetector
from manga_ar.detect.reading_order import assign_reading_order
from manga_ar.metrics import cer, mask_iou, match_boxes, precision_recall
from manga_ar.models.manager import ModelManager
from manga_ar.schemas import Flag, RegionType

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("MANGAAR_CACHE_DIR", ROOT / ".cache"))


def _pages(seeds: range) -> list[synth.SynthPage]:
    out: list[synth.SynthPage] = []
    for s in seeds:
        out += [synth.basic_page(lang, s, CACHE) for lang in ("ja", "ko", "zh")]
        out.append(synth.basic_page("ja", s, CACHE, vertical=False))
        out.append(synth.variety_page(("ja", "ko", "zh")[s % 3], s, CACHE))
        out.append(synth.texture_page(s, CACHE, ("ja", "ko", "zh")[s % 3]))
    out += [
        synth.furigana_page(0, CACHE),
        synth.symbols_page("ja", 0, CACHE),
        synth.adversarial_gutter_page(0, CACHE),
        synth.two_block_bubble_page(0, CACHE),
        synth.demo_page(CACHE),
    ]
    return out


def section_detection(
    pages: list[synth.SynthPage], lines: list[str], timings: dict[str, list[float]]
) -> dict[str, list]:  # type: ignore[type-arg]
    cfg = load_config(environ={})
    det, seg = ClassicalDetector(cfg.detect), FloodBubbleSegmenter(cfg.detect)
    n_t = n_p = n_m = 0
    ious: list[float] = []
    types_ok = types_n = 0
    order_ok = 0
    segmented: dict[str, list] = {}  # type: ignore[type-arg]
    for page in pages:
        t0 = time.perf_counter()
        blocks = det.detect(page.image)
        t1 = time.perf_counter()
        regions = seg.segment(page.image, blocks, page.name)
        regions = assign_reading_order(
            [r for r in regions if r.type != RegionType.SFX], page.image, page.reading_order
        )
        t2 = time.perf_counter()
        timings["detect"].append(t1 - t0)
        timings["segment+order"].append(t2 - t1)
        segmented[page.name] = regions
        m = match_boxes([g.text_bbox for g in page.regions], [r.bbox for r in regions], 0.5)
        n_t, n_p, n_m = n_t + len(page.regions), n_p + len(regions), n_m + len(m)
        h, w = page.image.shape[:2]
        for gi, rj in m:
            g, r = page.regions[gi], regions[rj]
            types_n += 1
            types_ok += g.type == r.type.value
            if (
                g.bubble_mask is not None
                and r.bubble_mask is not None
                and Flag.LEAK_FALLBACK not in r.flags
            ):
                ious.append(mask_iou(g.bubble_mask.to_full(h, w), r.bubble_mask.to_full(h, w)))
        order = [gi for gi, rj in sorted(m, key=lambda t: regions[t[1]].reading_order)]
        order_ok += order == sorted(order) and len(m) == len(page.regions)
    precision, recall = precision_recall(n_t, n_p, n_m)
    lines += [
        "## Detection & segmentation (classical detector)",
        "",
        "| metric | value | gate |",
        "|---|---|---|",
        f"| text-region precision @IoU0.5 | {precision:.3f} ({n_m}/{n_p}) | ≥ 0.90 |",
        f"| text-region recall @IoU0.5 | {recall:.3f} ({n_m}/{n_t}) | ≥ 0.95 |",
        f"| mean bubble-mask IoU | {np.mean(ious):.3f} (min {np.min(ious):.3f}, n={len(ious)}) "
        "| ≥ 0.85 |",
        f"| region type accuracy | {types_ok / max(1, types_n):.3f} ({types_ok}/{types_n}) | — |",
        f"| reading order exact | {order_ok / len(pages):.3f} ({order_ok}/{len(pages)} pages) "
        "| ≥ 0.95 |",
        "",
    ]
    return segmented


def section_ocr(
    pages: list[synth.SynthPage],
    segmented: dict[str, list],  # type: ignore[type-arg]
    lines: list[str],
    timings: dict[str, list[float]],
) -> None:
    from manga_ar.ocr.factory import build_router

    cfg = load_config(environ={"MANGAAR_CACHE_DIR": str(CACHE)})
    router = build_router(cfg, ModelManager(CACHE), "cpu")
    errors: dict[tuple[str, str], list[float]] = defaultdict(list)
    engines: dict[str, int] = defaultdict(int)
    for page in pages:
        if page.name.startswith(("texture", "demo")):
            continue
        regions = segmented[page.name]
        t0 = time.perf_counter()
        router.recognize_all(page.image, regions, page.lang)
        timings["ocr"].append(time.perf_counter() - t0)
        for gi, rj in match_boxes([g.text_bbox for g in page.regions], [r.bbox for r in regions]):
            g, r = page.regions[gi], regions[rj]
            errors[(page.lang, "vertical" if g.vertical else "horizontal")].append(
                cer(g.text, r.ocr.text if r.ocr else "")
            )
            engines[r.ocr.engine if r.ocr else "none"] += 1
    lines += [
        "## OCR (character error rate)",
        "",
        "| language | orientation | regions | CER | gate |",
        "|---|---|---|---|---|",
    ]
    for (lang, orient), vals in sorted(errors.items()):
        lines.append(f"| {lang} | {orient} | {len(vals)} | {np.mean(vals):.3f} | ≤ 0.10 |")
    lines += ["", "Engines used: " + ", ".join(f"{k}={v}" for k, v in sorted(engines.items())), ""]


def section_inpaint(
    pages: list[synth.SynthPage],
    segmented: dict[str, list],  # type: ignore[type-arg]
    lines: list[str],
) -> None:
    from manga_ar.inpaint.lama import LamaInpainter
    from manga_ar.inpaint.strategy import RegionInpainter

    rows = []
    for preset in ("fast", "balanced", "quality"):
        cfg = load_config(overrides={"preset": preset}, environ={})
        lama = LamaInpainter(ModelManager(CACHE)) if cfg.inpaint.use_lama else None
        from manga_ar.inpaint.residual import ResidualChecker

        checker = ResidualChecker(ClassicalDetector(cfg.detect))
        inp = RegionInpainter(cfg.inpaint, lama, checker)
        methods: dict[str, int] = defaultdict(int)
        outside = residual = n = 0
        elapsed = 0.0
        for page in pages:
            regions = [replace(r, flags=set(r.flags)) for r in segmented[page.name]]
            t0 = time.perf_counter()
            clean = inp.inpaint_page(page.image, regions)
            elapsed += time.perf_counter() - t0
            h, w = page.image.shape[:2]
            union = np.zeros((h, w), bool)
            gray = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY)
            for r in regions:
                if r.inpaint_mask is not None:
                    union |= r.inpaint_mask.to_full(h, w)
                if r.inpaint_method:
                    methods[r.inpaint_method] += 1
                if r.inpaint_method not in (None, "none"):
                    n += 1
                    box = r.bubble_mask.bbox if r.bubble_mask is not None else r.bbox
                    sub = gray[box.y0 : box.y1, box.x0 : box.x1]
                    zone = (
                        r.bubble_mask.data
                        if r.bubble_mask is not None
                        else np.ones(sub.shape, bool)
                    )
                    inside = cv2.erode(zone.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
                    if r.type != RegionType.FREE_TEXT:
                        dev = np.abs(sub.astype(int) - cv2.medianBlur(sub, 15)) > 30
                        residual += bool((dev & inside).any())
                    else:  # measured for every preset (not only when residual_check is on)
                        residual += checker.has_residual(clean, r)
            outside += int(((clean != page.image).any(axis=2) & ~union).sum())
        rows.append((preset, dict(methods), outside, residual, n, elapsed / len(pages)))
    lines += [
        "## Inpainting",
        "",
        "| preset | methods | px changed outside masks | regions with "
        "residual text | mean time / page |",
        "|---|---|---|---|---|",
    ]
    for preset, methods, outside, residual, n, t in rows:
        lines.append(
            f"| {preset} | {methods} | {outside} | {residual}/{n} "
            f"({1 - residual / max(1, n):.1%} clean) | {t * 1000:.0f} ms |"
        )
    lines += [
        "",
        "Residual text is measured two ways: on bubble/caption interiors, any pixel deviating "
        "> 30 levels from a 15 px median-filtered background (catches single-pixel remnants); "
        "on free text, by re-running the classical detector on the cleaned crop. The latter "
        "shares the detector's blind spot: glyphs drawn straight onto dense 1-px hatching "
        "without a halo can be missed by detection *and* by this check. Visual inspection "
        "found such a surviving glyph on the no-halo hatching fixture (DECISIONS D-019); "
        "`detect.detector: hybrid` recovers these pages.",
        "",
    ]


class FakeClock:
    """Instant clock: ``sleep`` advances simulated time."""

    def __init__(self) -> None:
        self.now = 1000.0

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(0.0, seconds)


class _FlakyProvider:
    """Fake provider failing with a given probability (seeded); 429s carry Retry-After."""

    def __init__(self, name: str, fail_rate: float, rate_limit_share: float, seed: int) -> None:
        import random as _random

        self.name, self.network = name, True
        self.fail_rate, self.rate_limit_share = fail_rate, rate_limit_share
        self.rng = _random.Random(seed)
        self.calls = 0

    def available(self) -> bool:
        return True

    def supports(self, src: str, tgt: str) -> bool:
        return True

    def translate(self, text: str, src: str, tgt: str) -> str:
        from manga_ar.errors import ProviderError, RateLimitError
        from manga_ar.translate.batching import decode, encode

        self.calls += 1
        if self.rng.random() < self.fail_rate:
            if self.rng.random() < self.rate_limit_share:
                raise RateLimitError("429", retry_after=3.0)
            raise ProviderError("503")
        segments = decode(text, text.count("[")) if text.startswith("[1]") else None
        if segments is not None:
            return encode([f"ترجمة {len(t)}" for t in segments])
        return f"ترجمة {len(text)}"


def section_typeset(
    pages: list[synth.SynthPage],
    segmented: dict[str, list],  # type: ignore[type-arg]
    lines: list[str],
    timings: dict[str, list[float]],
) -> None:
    """Lay out corpus strings (cycled, short to very long) into every detected region."""
    import json

    from manga_ar.schemas import PageDocument, TranslationResult
    from manga_ar.typeset.fonts import FontRegistry
    from manga_ar.typeset.layout import Typesetter
    from manga_ar.typeset.page import typeset_page

    corpus = json.loads((ROOT / "tests" / "data" / "arabic_corpus.json").read_text("utf-8"))
    texts = [item["text"] for item in corpus["canonical"]]
    cfg = load_config(environ={})
    ts = Typesetter(cfg.typeset, FontRegistry())
    n = overflow = outside_regions = ladder_used = 0
    sizes: list[float] = []
    k = 0
    for page in pages:
        h, w = page.image.shape[:2]
        regions = [replace(r, flags=set(r.flags)) for r in segmented[page.name]]
        for r in regions:
            r.translation = TranslationResult(provider="bench", text=texts[k % len(texts)])
            k += 1
        doc = PageDocument(source=page.name, width=w, height=h, regions=regions)
        t0 = time.perf_counter()
        out = typeset_page(doc, page.image, page.image, ts)
        timings["typeset"].append(time.perf_counter() - t0)
        for r in regions:
            p = out.placements.get(r.id)
            if p is None:
                continue
            n += 1
            sizes.append(p.size / h)
            ladder_used += bool(p.ladder)
            if Flag.OVERFLOW_RISK in r.flags:
                overflow += 1
                continue
            geom = p.geometry
            assert geom is not None
            allowed = np.zeros((h, w), np.uint8)
            b = geom.box.clip(w, h)
            allowed[b.y0 : b.y1, b.x0 : b.x1] = geom.mask[: b.height, : b.width]
            allowed = cv2.dilate(allowed, np.ones((5, 5), np.uint8)) > 0
            from manga_ar.typeset.render import render_layer

            ink = render_layer((h, w), [p])[..., 3] > 40
            outside_regions += bool((ink & ~allowed).any())
    per_region = np.sum(timings["typeset"]) / max(1, n) * 1000
    lines += [
        "## Typesetting (corpus strings cycled into detected regions)",
        "",
        "| metric | value | gate |",
        "|---|---|---|",
        f"| regions typeset | {n} | — |",
        f"| ink outside layout area (non-overflow regions) | {outside_regions} | 0 |",
        f"| OVERFLOW_RISK (hard floor) | {overflow} ({overflow / max(1, n):.1%}) | reported |",
        f"| regions needing ladder steps | {ladder_used} ({ladder_used / max(1, n):.1%}) | — |",
        f"| mean font size / page height | {np.mean(sizes):.4f} | — |",
        f"| mean time per region | {per_region:.1f} ms | < 50 ms (A13) |",
        "",
        "Texts are assigned regardless of bubble size (the 90-character sentence lands in "
        "small bubbles too), so OVERFLOW_RISK here is a stress figure, not a typical rate. "
        "ARVS L2–L7 results live in the test suite (DECISIONS D-030/D-031).",
        "",
    ]


def section_translation(lines: list[str]) -> None:
    """Mocked failover statistics (fake providers, fake clock: no network, no sleeping)."""
    from manga_ar.schemas import BBox, OcrResult, Region
    from manga_ar.translate.service import TranslationService

    cfg = load_config(environ={})
    lines += [
        "## Translation failover (mocked providers, fake clock)",
        "",
        "| scenario | regions | served by google / mymemory / local | untranslated | "
        "simulated wait |",
        "|---|---|---|---|---|",
    ]
    for label, rates in (
        ("healthy", (0.0, 0.0, 0.0)),
        ("google 30 % errors", (0.3, 0.0, 0.0)),
        ("google down, mymemory 50 %", (1.0, 0.5, 0.0)),
        ("all online down", (1.0, 1.0, 0.0)),
    ):
        providers = [
            _FlakyProvider("google", rates[0], 0.5, 1),
            _FlakyProvider("mymemory", rates[1], 0.5, 2),
            _FlakyProvider("local", rates[2], 0.0, 3),
        ]
        providers[2].network = False
        clock = FakeClock()
        svc = TranslationService(providers, cfg, None, None, clock)
        served: dict[str, int] = defaultdict(int)
        untranslated = total = 0
        for page in range(20):
            regions = [
                Region(
                    id=f"p{page}r{i}",
                    type=RegionType.BUBBLE,
                    bbox=BBox(0, 0, 1, 1),
                    reading_order=i,
                    ocr=OcrResult("x", f"テキスト{page}-{i}", lang="ja"),
                )
                for i in range(6)
            ]
            svc.translate_regions(regions, "ja")
            for r in regions:
                total += 1
                if r.translation is not None and r.translation.provider != "none":
                    served[r.translation.provider] += 1
                else:
                    untranslated += 1
        lines.append(
            f"| {label} | {total} | {served['google']} / {served['mymemory']} / "
            f"{served['local']} | {untranslated} | {clock.now - 1000.0:.0f} s |"
        )
    lines.append("")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs=2, type=int, default=(100, 104))
    parser.add_argument("--no-ocr", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "QUALITY_REPORT.md")
    args = parser.parse_args()
    pages = _pages(range(args.seeds[0], args.seeds[1]))
    timings: dict[str, list[float]] = defaultdict(list)
    lines = [
        "# Quality Report",
        "",
        f"Generated by `scripts/benchmark.py` — MangaAR {__version__}, "
        f"{platform.system()} {platform.machine()}, Python {platform.python_version()}, CPU only.",
        f"Held-out synthetic pages: seeds {args.seeds[0]}–{args.seeds[1] - 1} "
        f"({len(pages)} pages, {sum(len(p.regions) for p in pages)} ground-truth regions). "
        "Thresholds were calibrated on seeds 0–2 (see DECISIONS.md).",
        "",
    ]
    segmented = section_detection(pages, lines, timings)
    if not args.no_ocr:
        section_ocr(pages, segmented, lines, timings)
    section_inpaint(pages, segmented, lines)
    section_typeset(pages, segmented, lines, timings)
    section_translation(lines)
    lines += ["## Stage timings (CPU)", "", "| stage | mean / page | max |", "|---|---|---|"]
    for stage, vals in timings.items():
        lines.append(f"| {stage} | {np.mean(vals) * 1000:.0f} ms | {np.max(vals) * 1000:.0f} ms |")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    lines += ["", f"Peak resident memory of the benchmark process: {peak:.0f} MB.", ""]
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
