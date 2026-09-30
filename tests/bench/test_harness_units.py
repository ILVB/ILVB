"""Harness scoring and aggregation on hand-built pages (no models)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from benchmarks import rle, run_benchmark
from benchmarks.evaluate import score_page
from benchmarks.schema import (
    GtPage,
    GtRegion,
    PageResult,
    PredRegion,
    StageStats,
    TypesetReport,
)
from benchmarks.summary import summarize

H, W = 60, 80
REF = "مرحبا بك في بيتنا الجميل"  # >= 4 tokens: corpus BLEU needs 4-grams


def _mask(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


def _region(rid: str, kind: str, box: tuple[int, int, int, int], text: str) -> GtRegion:
    x0, y0, x1, y1 = box
    return GtRegion(
        region_id=rid, category="flat_white", type=kind, lang="ja", text=text,  # type: ignore[arg-type]
        text_polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)], text_mask=rle.encode(_mask(*box)),
        safe_mask=rle.encode(_mask(x0 - 5, y0 - 5, x1 + 5, y1 + 5)), reading_order=int(rid[-1]),
        references_ar=[] if kind == "sfx" else [REF], reference_kind="silver",
    )  # fmt: skip


def _page() -> GtPage:
    regions = [_region("p-r1", "dialogue", (10, 10, 30, 20), "こんにちは"),
               _region("p-r2", "sfx", (50, 30, 70, 50), "ドン")]  # fmt: skip
    return GtPage(page_id="p", series_id="s", split="dev", categories=["flat_white"], lang="ja",
                  reading_direction="rtl", width=W, height=H, image="p.png", clean="p.clean.png",
                  seed=1, generator="test", regions=regions)  # fmt: skip


def _results(tmp: Path, original: np.ndarray, clean: np.ndarray) -> dict[str, PageResult]:
    erased = clean.copy()
    sfx = _mask(50, 30, 70, 50)
    erased[sfx] = original[sfx]  # SFX is not erased (not declared either)
    erased[0, 0] = 7  # one stray changed pixel outside the declared mask
    Image.fromarray(erased).save(tmp / "p.erase.erased.png")
    pred = PredRegion(region_id="x", polygon=[(10, 10), (30, 10), (30, 20), (10, 20)],
                      text="こんにちわ", reading_order=0,
                      erase_mask=rle.encode(_mask(8, 8, 32, 22)))  # fmt: skip
    ink = rle.encode(_mask(12, 12, 28, 18))
    return {
        "erase": PageResult(page_id="p", version="t", mode="erase", regions=[pred],
                            erased_image="p.erase.erased.png"),
        "typeset_gt": PageResult(page_id="p", version="t", mode="typeset_gt", typeset=[
            TypesetReport(region_id="p-r1", typeset=True, size_px=20.0, lines=["مرحبا بك"],
                          line_boxes=[[12, 12, 28, 18]], ink_mask=ink)]),
        "translate_gt": PageResult(page_id="p", version="t", mode="translate_gt",
                                   translations={"p-r1": REF}),
    }  # fmt: skip


def _images() -> tuple[np.ndarray, np.ndarray]:
    clean = np.full((H, W, 3), 255, np.uint8)
    original = clean.copy()
    original[_mask(10, 10, 30, 20) | _mask(50, 30, 70, 50)] = 0
    return original, clean


def _row(tmp: Path) -> tuple[GtPage, dict[str, Any]]:
    original, clean = _images()
    gt = _page()
    return gt, score_page(gt, "flat_white", _results(tmp, original, clean), tmp, original, clean)


def test_score_page(tmp_path: Path) -> None:
    _gt, row = _row(tmp_path)
    det = row["detection"]
    assert (det["tp"], det["gt"], det["pred"]) == (1, 2, 1)
    assert [(r["edits"], r["length"]) for r in det["ocr"]] == [(1, 5), (2, 2)]  # sfx unmatched
    assert row["inpaint"]["changed_outside"] == 1
    assert [r["region_id"] for r in row["inpaint"]["regions"]] == ["p-r1"]  # sfx excluded
    assert row["typeset"][0]["ink_outside"] == 0 and row["typeset"][0]["above_floor"]
    assert row["translation"] == [{"region_id": "p-r1", "hyp": REF}]


def test_failed_modes_score_as_untouched_page(tmp_path: Path) -> None:
    original, clean = _images()
    failed = {m: PageResult(page_id="p", version="t", mode=m, errors=["Boom: x"])  # type: ignore[arg-type]
              for m in ("erase", "typeset_gt", "translate_gt")}  # fmt: skip
    row = score_page(_page(), "flat_white", failed, tmp_path, original, clean)
    assert set(row["errors"]) == {"erase", "typeset_gt", "translate_gt"}
    assert row["detection"]["tp"] == 0 and row["inpaint"]["regions"][0]["residual"]
    assert row["typeset"] == [{"region_id": "p-r1", "typeset": False}]
    assert row["translation"] == [{"region_id": "p-r1", "hyp": ""}]


def test_cli_refuses_sealed_split(capsys: pytest.CaptureFixture[str]) -> None:
    assert run_benchmark.main(["--version", "baseline", "--split", "test"]) == 2
    assert "sealed" in capsys.readouterr().err


def test_raw_digest_is_reproducible_and_tracks_content(tmp_path: Path) -> None:
    img = np.zeros((4, 4, 3), np.uint8)
    Image.fromarray(img).save(tmp_path / "p.png")
    res = PageResult(page_id="p", version="t", mode="erase", erased_image="p.png")
    first = run_benchmark.raw_digest(tmp_path, [res])
    timed = res.model_copy(update={"stages": {"ocr": StageStats(seconds=9.0, peak_rss_mb=1.0)},
                                   "code_origin": "/elsewhere"})  # fmt: skip
    assert run_benchmark.raw_digest(tmp_path, [timed]) == first  # timings are not content
    Image.fromarray(img).save(tmp_path / "p.png", compress_level=1)
    assert run_benchmark.raw_digest(tmp_path, [res]) == first  # encoder settings neither
    img[0, 0] = 1
    Image.fromarray(img).save(tmp_path / "p.png")
    assert run_benchmark.raw_digest(tmp_path, [res]) != first  # one pixel is
    edited = res.model_copy(update={"errors": ["x"]})
    assert run_benchmark.raw_digest(tmp_path, [edited]) != run_benchmark.raw_digest(tmp_path, [res])


def test_summary(tmp_path: Path) -> None:
    gt, row = _row(tmp_path)
    refs = {("p", r.region_id): r.references_ar for r in gt.regions}
    s = summarize([row], refs, "silver")
    assert s["ocr"]["cer"] == pytest.approx(1 / 5) and s["ocr"]["cer_sfx"] == 1.0
    assert s["ocr"]["precision"] == 1.0 and s["ocr"]["recall"] == 0.5
    assert s["inpaint"]["pages_background_intact"] == 0.0
    assert s["typeset"]["inside_safe_rate"] == 1.0
    assert s["translation"]["bleu"] == pytest.approx(100)
    assert s["translation"]["label"] == "SILVER-REFERENCE, comparative only"


def test_baseline_refuses_engine_profiles() -> None:
    with pytest.raises(ValueError, match="engine"):
        run_benchmark.run("baseline", "dev", profile="v2-offline")


def test_content_digests_ignore_the_implementation(tmp_path: Path) -> None:
    from benchmarks.compare_runs import differences

    img = np.zeros((4, 4, 3), np.uint8)
    Image.fromarray(img).save(tmp_path / "p.png")
    a = PageResult(page_id="p", version="baseline", mode="erase", erased_image="p.png")
    b = a.model_copy(update={"version": "candidate", "code_origin": "/src"})
    da, db = (
        run_benchmark.content_digests(tmp_path, [a]),
        run_benchmark.content_digests(tmp_path, [b]),
    )
    assert differences(da, db) == []
    c = a.model_copy(update={"errors": ["boom"]})
    assert differences(da, run_benchmark.content_digests(tmp_path, [c])) == ["p.erase"]
    assert differences(da, {}) == ["p.erase"]


def test_ocr_by_category_groups(tmp_path: Path) -> None:
    from benchmarks.summary import ocr_by_category

    _gt, row = _row(tmp_path)
    cats = ocr_by_category([row])
    assert cats["flat_white"]["cer"] == pytest.approx(1 / 5)
    assert cats["G-OCR-1:standard_bubbles"]["pages"] == 1 and "G-OCR-1:screentone" not in cats


def test_modes_that_did_not_run_are_left_out(tmp_path: Path) -> None:
    original, clean = _images()
    gt = _page()
    only = {"erase": _results(tmp_path, original, clean)["erase"]}
    row = score_page(gt, "flat_white", only, tmp_path, original, clean)
    assert "detection" in row and "typeset" not in row and "translation" not in row
    s = summarize([row], {}, "silver")
    assert "ocr" in s and "typeset" not in s and "translation" not in s
    assert s["ocr"]["mask_iou"] == 0.0  # the prediction carries no text mask
    erase = only["erase"]
    pred = erase.regions[0].model_copy(update={"text_mask": rle.encode(_mask(10, 10, 30, 20))})
    masked = {"erase": erase.model_copy(update={"regions": [pred]})}
    row = score_page(gt, "flat_white", masked, tmp_path, original, clean)
    assert row["detection"]["mask_iou"] == [1.0]  # identical to the ground-truth mask
