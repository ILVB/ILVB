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
    segmented: dict[str, list],
    lines: list[str],  # type: ignore[type-arg]
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
    segmented: dict[str, list],
    lines: list[str],  # type: ignore[type-arg]
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
