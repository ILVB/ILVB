"""0.2.2: both adapters run the same inputs through each stage (real models)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from benchmarks import rle
from benchmarks.adapters.client import run_jobs
from benchmarks.schema import GtPage, GtRegion

pytestmark = pytest.mark.integration
pytest.importorskip("rapidocr_onnxruntime")


def _gt_from_demo(tmp_path: Path, cache_dir: Path) -> GtPage:
    from manga_ar import synth

    page = synth.demo_page(cache_dir)
    h, w = page.image.shape[:2]
    Image.fromarray(page.image).save(tmp_path / "page.png")
    Image.fromarray(np.full_like(page.image, 255)).save(tmp_path / "clean.png")
    regions = []
    for g in page.regions:
        text = g.text_mask.to_full(h, w)
        bubble = g.bubble_mask.to_full(h, w) if g.bubble_mask is not None else None
        regions.append(
            GtRegion(
                region_id=g.id,
                category="flat",
                type="narration" if g.type == "narration" else "dialogue",
                lang="zh",
                text=g.text,
                text_polygon=[(g.text_bbox.x0, g.text_bbox.y0), (g.text_bbox.x1, g.text_bbox.y0),
                              (g.text_bbox.x1, g.text_bbox.y1), (g.text_bbox.x0, g.text_bbox.y1)],
                text_mask=rle.encode(text),
                bubble_mask=rle.encode(bubble) if bubble is not None else None,
                safe_mask=rle.encode(bubble if bubble is not None else text),
                reading_order=g.reading_order + 1,
            )
        )  # fmt: skip
    gt = GtPage(
        page_id="demo", series_id="S", split="dev", categories=["flat"], lang="zh",
        reading_direction="ltr", width=w, height=h, image="page.png", clean="clean.png",
        seed=7, generator="manga_ar.synth.demo_page", regions=regions,
    )  # fmt: skip
    (tmp_path / "gt.json").write_text(gt.model_dump_json(), encoding="utf-8")
    return gt


def test_same_inputs_both_versions(tmp_path: Path, cache_dir: Path) -> None:
    try:
        gt = _gt_from_demo(tmp_path, cache_dir)
    except Exception as exc:  # noqa: BLE001 - the CJK fixture font may be missing
        pytest.skip(f"demo page unavailable: {exc}")
    cfg = {"input.source_lang": "zh", "translate.providers": ["tm"], "runtime.offline": True,
           "translate.cache": False, "inpaint.use_lama": False}  # fmt: skip
    texts = {r.region_id: "نص تجريبي" for r in gt.regions}
    jobs = [
        {"mode": m, "page_id": "demo", "image": str(tmp_path / "page.png"),
         "clean": str(tmp_path / "clean.png"), "gt": str(tmp_path / "gt.json"), "lang": "zh",
         "texts": texts, "config": cfg, "tm": {r.text: "نص" for r in gt.regions}}
        for m in ("detect_ocr", "erase", "typeset_gt", "translate_gt")
    ]  # fmt: skip
    results = {}
    for impl in ("baseline", "candidate"):
        results[impl] = run_jobs(impl, jobs, tmp_path / impl)
        for res in results[impl]:
            assert not res.errors, res.errors
            assert res.version == impl and res.stages
    base, cand = results["baseline"], results["candidate"]
    assert ".baseline" in base[0].code_origin and ".baseline" not in cand[0].code_origin
    assert {r.text for r in base[0].regions} == {r.text for r in gt.regions}
    assert (tmp_path / "baseline" / base[1].erased_image).is_file()
    assert all(t.typeset and t.ink_mask for t in base[2].typeset)
    assert base[3].translations == {r.region_id: "نص" for r in gt.regions}
    # identical code today → identical outputs (determinism across processes)
    assert [r.model_dump() for r in base[0].regions] == [r.model_dump() for r in cand[0].regions]
