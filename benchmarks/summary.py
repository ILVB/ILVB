"""Aggregate per-page metric rows into the numbers reports and the gate read."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from benchmarks.metrics import stats, translation
from benchmarks.metrics.detection import DetectionCounts

Row = Mapping[str, Any]
Refs = Mapping[tuple[str, str], list[str]]


def _ratio(num: float, den: float) -> float | None:
    return num / den if den else None


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def cer(rows: Sequence[Row], sfx: bool = False) -> float | None:
    regs = [r for row in rows for r in row["detection"]["ocr"] if (r["type"] == "sfx") == sfx]
    return _ratio(sum(r["edits"] for r in regs), sum(r["length"] for r in regs))


def ssim_mean(rows: Sequence[Row]) -> float | None:
    return _mean([r["ssim"] for row in rows for r in row["inpaint"]["regions"]])


def _ci(rows: Sequence[Row], fn: Callable[[Sequence[Row]], float | None]) -> list[float] | None:
    if not rows:
        return None

    def stat(sample: Sequence[object]) -> float:
        value = fn([r for r in sample if isinstance(r, Mapping)])
        return float("nan") if value is None else value

    lo, hi = stats.bootstrap_ci(list(rows), stat)
    return [lo, hi]


def ocr_summary(rows: Sequence[Row]) -> dict[str, Any]:
    det = DetectionCounts(0, 0, 0)
    for row in rows:
        d = row["detection"]
        det = det + DetectionCounts(d["tp"], d["gt"], d["pred"])
    words = [
        r for row in rows for r in row["detection"]["ocr"] if "words" in r and r["type"] != "sfx"
    ]
    by_lang: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        by_lang[row["lang"]].append(row)
    return {
        "cer": cer(rows), "cer_ci95": _ci(rows, cer), "cer_sfx": cer(rows, sfx=True),
        "cer_by_lang": {k: cer(v) for k, v in sorted(by_lang.items())},
        "wer": _ratio(sum(r["word_edits"] for r in words), sum(r["words"] for r in words)),
        "precision": det.precision, "recall": det.recall, "f1": det.f1,
        "mask_iou": _mean([v for row in rows for v in row["detection"].get("mask_iou", [])]),
        "order_accuracy": _ratio(sum(r["detection"]["order_hits"] for r in rows),
                                 sum(r["detection"]["order_total"] for r in rows)),
    }  # fmt: skip


# G-OCR-1 categories -> synthetic_v1 categories (docs/METRICS.md). The remaining synthetic
# categories (dark_bubble, text_on_art, sfx, tiny_text) are reported but not constrained.
GOCR_CATEGORIES: dict[str, tuple[str, ...]] = {
    "standard_bubbles": ("flat_white", "tails_overlap"),
    "vertical_text": ("vertical_ja",),
    "low_contrast": ("low_contrast",),
    "screentone": ("screentone",),
    "stylized_fonts": ("stylized",),
}


def ocr_by_category(rows: Sequence[Row]) -> dict[str, dict[str, Any]]:
    """CER / WER / detection per synthetic category and per G-OCR-1 category group."""
    groups: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        groups[row["category"]].append(row)
    for name, cats in GOCR_CATEGORIES.items():
        groups[f"G-OCR-1:{name}"] = [r for r in rows if r["category"] in cats]
    out = {}
    for name, members in sorted(groups.items()):
        if not members:
            continue
        s = ocr_summary(members)
        out[name] = {"pages": len(members), "cer": s["cer"], "wer": s["wer"],
                     "precision": s["precision"], "recall": s["recall"], "f1": s["f1"]}  # fmt: skip
    return out


def inpaint_summary(rows: Sequence[Row]) -> dict[str, Any]:
    regs = [r for row in rows for r in row["inpaint"]["regions"]]
    by_cat: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        by_cat[row["category"]].append(row)
    return {
        "pages_background_intact": _ratio(
            sum(row["inpaint"]["changed_outside"] == 0 for row in rows), len(rows)),
        "changed_outside_px": sum(row["inpaint"]["changed_outside"] for row in rows),
        "overreach_px": sum(row["inpaint"]["overreach"] for row in rows),
        "ssim": ssim_mean(rows), "ssim_ci95": _ci(rows, ssim_mean),
        "psnr": _mean([r["psnr"] for r in regs]),
        "ssim_by_category": {k: ssim_mean(v) for k, v in sorted(by_cat.items())},
        "residual_rate": _ratio(sum(r["residual"] for r in regs), len(regs)),
        "regions": len(regs),
    }  # fmt: skip


def typeset_summary(rows: Sequence[Row]) -> dict[str, Any]:
    regs = [r for row in rows for r in row["typeset"]]
    done = [r for r in regs if r["typeset"]]
    ragged = [r["raggedness"] for r in done if r["raggedness"] is not None]
    multi = [r for r in done if r["lines"] >= 2]
    return {
        "regions": len(regs), "typeset_rate": _ratio(len(done), len(regs)),
        "inside_safe_rate": _ratio(sum(r["ink_outside"] == 0 for r in done), len(regs)),
        "above_floor_rate": _ratio(sum(bool(r["above_floor"]) for r in done), len(regs)),
        "raggedness": _mean(ragged),
        "orphan_rate": _ratio(sum(r["orphan"] for r in multi), len(multi)),
        "hyphen_breaks": sum(r["hyphen_breaks"] for r in done),
        "overflow_rate": _ratio(sum(bool(r["overflow"]) for r in done), len(regs)),
    }  # fmt: skip


def segments(rows: Sequence[Row], refs: Refs) -> tuple[list[str], list[list[str]]]:
    hyps, rr = [], []
    for row in rows:
        for seg in row["translation"]:
            hyps.append(seg["hyp"])
            rr.append(refs[(row["page_id"], seg["region_id"])])
    return hyps, rr


def translation_summary(rows: Sequence[Row], refs: Refs, kind: str) -> dict[str, Any]:
    hyps, rr = segments(rows, refs)
    if not hyps:
        return {"segments": 0}
    return {
        "segments": len(hyps), "translated_rate": sum(bool(h.strip()) for h in hyps) / len(hyps),
        "bleu": translation.bleu(hyps, rr), "chrf_pp": translation.chrf_pp(hyps, rr),
        "meteor": translation.meteor(hyps, rr), "reference_kind": kind,
        "label": "SILVER-REFERENCE, comparative only" if kind == "silver" else kind.upper(),
    }  # fmt: skip


def perf_summary(rows: Sequence[Row]) -> dict[str, Any]:
    per_stage: dict[str, list[float]] = defaultdict(list)
    rss: list[float] = []
    for row in rows:
        for mode_stats in row["stages"].values():
            for stage, s in mode_stats.items():
                per_stage[stage].append(s["seconds"])
                rss.append(s["peak_rss_mb"])
    pipeline = [sum(s["seconds"] for s in row["stages"].get("erase", {}).values()) for row in rows]
    return {
        "stage_seconds": {k: stats.percentiles(v) for k, v in sorted(per_stage.items())},
        "erase_page_seconds": stats.percentiles(pipeline),
        "peak_rss_mb": max(rss) if rss else None,
    }


def summarize(rows: Sequence[Row], refs: Refs, reference_kind: str) -> dict[str, Any]:
    errors: dict[str, int] = defaultdict(int)
    for row in rows:
        for mode in row["errors"]:
            errors[mode] += 1

    def ran(key: str) -> bool:
        return bool(rows) and all(key in row for row in rows)

    out: dict[str, Any] = {"pages": len(rows), "errors": dict(sorted(errors.items()))}
    if ran("detection"):
        out |= {"ocr": ocr_summary(rows), "ocr_by_category": ocr_by_category(rows),
                "inpaint": inpaint_summary(rows)}  # fmt: skip
    if ran("typeset"):
        out["typeset"] = typeset_summary(rows)
    if ran("translation"):
        out["translation"] = translation_summary(rows, refs, reference_kind)
    out["perf"] = perf_summary(rows)
    return out
