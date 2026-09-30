"""Score raw adapter results against ground truth: one metric row per page.

Rows hold numbers and short strings only (no images), so a frozen result file stays small
and every summary can be recomputed from it (PD-2).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from benchmarks import rle
from benchmarks.metrics import inpaint, typeset
from benchmarks.metrics.detection import box_of, match, order_hits
from benchmarks.metrics.text import char_edits, word_edits
from benchmarks.schema import GtPage, PageResult

U8 = npt.NDArray[np.uint8]
WORD_LANGS = {"en", "ko"}


def load_rgb(path: Path) -> U8:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def score_ocr(gt: GtPage, result: PageResult) -> dict[str, Any]:
    """Detection counts, reading order and end-to-end OCR edits (unmatched GT = deleted)."""
    preds = result.regions
    pairs = match([box_of(r.text_polygon) for r in gt.regions], [box_of(p.polygon) for p in preds])
    by_gt = {gi: pi for gi, pi, _ in pairs}
    ocr = []
    for gi, g in enumerate(gt.regions):
        if not g.text:
            continue  # illegible in the gold set: excluded from CER
        hyp = preds[by_gt[gi]].text if gi in by_gt else ""
        c = char_edits(g.text, hyp)
        row = {"region_id": g.region_id, "type": g.type, "matched": gi in by_gt,
               "edits": c.edits, "length": c.length}  # fmt: skip
        if gt.lang in WORD_LANGS:
            w = word_edits(g.text, hyp)
            row |= {"word_edits": w.edits, "words": w.length}
        ocr.append(row)
    shape = (gt.height, gt.width)
    mask_ious = []
    for gi, pi, _ in pairs:
        g, pred = gt.regions[gi], preds[pi]
        if g.type == "sfx":
            continue
        truth = rle.decode(g.text_mask, shape)
        found = rle.decode(pred.text_mask, shape) if pred.text_mask is not None else None
        union = int((truth | found).sum()) if found is not None else int(truth.sum())
        inter = int((truth & found).sum()) if found is not None else 0
        mask_ious.append(inter / union if union else 0.0)
    order_pairs = [(gt.regions[gi].reading_order, preds[pi].reading_order)
                   for gi, pi, _ in pairs if gt.regions[gi].type != "sfx"]  # fmt: skip
    hits, total = order_hits(order_pairs)
    return {"tp": len(pairs), "gt": len(gt.regions), "pred": len(preds),
            "order_hits": hits, "order_total": total, "mask_iou": mask_ious,
            "ocr": ocr}  # fmt: skip


def score_inpaint(
    gt: GtPage, result: PageResult, original: U8, erased: U8, clean: U8
) -> dict[str, Any]:
    shape = (gt.height, gt.width)
    declared = np.zeros(shape, np.bool_)
    for p in result.regions:
        if p.erase_mask is not None:
            declared |= rle.decode(p.erase_mask, shape)
    lettering = np.zeros(shape, np.bool_)
    regions = []
    for g in gt.regions:
        text = rle.decode(g.text_mask, shape)
        lettering |= text
        if g.type == "sfx":
            continue
        ssim, psnr = inpaint.ssim_psnr(erased, clean, inpaint.crop_box(text, 8))
        probe_box = inpaint.crop_box(text, 4)
        probe = inpaint.probe_residual(erased, probe_box) and not inpaint.probe_residual(
            clean, probe_box
        )  # a probe that also fires on the clean page is not evidence of residual text
        pixel = inpaint.pixel_residual(erased, clean, text)
        regions.append({"region_id": g.region_id, "ssim": ssim, "psnr": psnr,
                        "residual": probe or pixel, "residual_probe": probe,
                        "residual_pixel": pixel})  # fmt: skip
    return {"changed_outside": inpaint.changed_outside(original, erased, declared),
            "overreach": inpaint.overreach(declared, lettering), "regions": regions}  # fmt: skip


def score_typeset(gt: GtPage, result: PageResult, requested: set[str]) -> list[dict[str, Any]]:
    shape = (gt.height, gt.width)
    safe = {g.region_id: g.safe_mask for g in gt.regions}
    rows = []
    for rep in result.typeset:
        if rep.region_id not in requested:
            continue
        if not rep.typeset or rep.ink_mask is None:
            rows.append({"region_id": rep.region_id, "typeset": False})
            continue
        mask = rle.decode(safe[rep.region_id], shape)
        rows.append({
            "region_id": rep.region_id, "typeset": True, "size_px": rep.size_px,
            "ink_outside": typeset.ink_outside(rle.decode(rep.ink_mask, shape), mask),
            "above_floor": typeset.above_floor(rep.size_px, gt.height),
            "raggedness": typeset.raggedness(rep.line_boxes, mask),
            "orphan": typeset.is_orphan(rep.lines), "lines": len(rep.lines),
            "hyphen_breaks": typeset.arabic_hyphen_breaks(rep.lines), "overflow": rep.overflow,
        })  # fmt: skip
    missing = requested - {r["region_id"] for r in rows}
    rows += [{"region_id": rid, "typeset": False} for rid in sorted(missing)]
    return rows


def typeset_texts(gt: GtPage) -> dict[str, str]:
    """Identical Arabic input for every version: the first reference of each text region."""
    return {g.region_id: g.references_ar[0] for g in gt.regions
            if g.type != "sfx" and g.references_ar}  # fmt: skip


def score_translation(gt: GtPage, result: PageResult) -> list[dict[str, str]]:
    return [{"region_id": g.region_id, "hyp": result.translations.get(g.region_id, "")}
            for g in gt.regions if g.type != "sfx" and g.references_ar]  # fmt: skip


def score_page(gt: GtPage, category: str, results: dict[str, PageResult], raw_dir: Path,
               original: U8, clean: U8) -> dict[str, Any]:  # fmt: skip
    """Metric groups for the modes that were run. A failed mode is scored as a page left
    untouched (no detections, nothing erased, nothing typeset or translated), never skipped."""
    row: dict[str, Any] = {"page_id": gt.page_id, "category": category, "lang": gt.lang,
                           "errors": {}, "stages": {}}  # fmt: skip
    for mode, res in results.items():
        row["stages"][mode] = {k: v.model_dump() for k, v in res.stages.items()}
        if res.errors:
            row["errors"][mode] = res.errors[0]
    empty = PageResult(page_id=gt.page_id, version="", mode="erase")
    if "erase" in results:  # modes that were not run at all are left out, not failed
        erase = results["erase"]
        ok = not erase.errors and erase.erased_image is not None
        erase = erase if ok else empty
        erased = load_rgb(raw_dir / erase.erased_image) if erase.erased_image else original
        row["detection"] = score_ocr(gt, erase)
        row["inpaint"] = score_inpaint(gt, erase, original, erased, clean)
    if "typeset_gt" in results:
        ts = results["typeset_gt"]
        row["typeset"] = score_typeset(gt, ts if not ts.errors else empty, set(typeset_texts(gt)))
    if "translate_gt" in results:
        tr = results["translate_gt"]
        row["translation"] = score_translation(gt, tr if not tr.errors else empty)
    return row
