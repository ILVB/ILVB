"""0.3.5: gold-data validation and ingestion (templates as fixtures)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from benchmarks.schema import GtPage
from tools import ingest_gold

TEMPLATES = Path(__file__).resolve().parents[2] / "benchmarks" / "data" / "templates"


def _gold(tmp_path: Path) -> Path:
    src = tmp_path / "gold"
    src.mkdir()
    for name in ("pages.csv", "regions.jsonl", "arabic_references.csv", "glossary.yaml",
                 "speakers.csv"):  # fmt: skip
        shutil.copy(TEMPLATES / f"{name}.template", src / name)
    (src / "images").mkdir()
    for name in ("S01_C001_P001.png", "S01_C001_P002.png"):
        Image.fromarray(np.full((800, 700, 3), 255, np.uint8)).save(src / "images" / name)
    return src


def test_templates_validate_and_ingest(tmp_path: Path) -> None:
    src = _gold(tmp_path)
    report, data = ingest_gold.validate(src)
    assert report.problems == []
    digests = ingest_gold.ingest(src, data)
    assert set(digests) == {"dev", "val", "test"}
    files = list((src / "ingested").rglob("*.gt.json"))
    assert len(files) == 2
    page = next(GtPage.model_validate_json(f.read_text(encoding="utf-8")) for f in files
                if "P001" in f.name)  # fmt: skip
    assert [r.type for r in page.regions] == ["dialogue", "sfx"]
    assert page.regions[0].reference_kind == "gold" and page.regions[0].references_ar


def test_problems_are_located(tmp_path: Path) -> None:
    src = _gold(tmp_path)
    lines = (src / "regions.jsonl").read_text(encoding="utf-8").splitlines()
    bad = json.loads(lines[0])
    bad["type"] = "speech"
    bad["text_polygon"] = [[0, 0], [1, 1]]
    (src / "regions.jsonl").write_text(json.dumps(bad) + "\n{broken\n", encoding="utf-8")
    report, _ = ingest_gold.validate(src)
    joined = "\n".join(report.problems)
    assert "regions.jsonl:1: type 'speech'" in joined
    assert "regions.jsonl:1: text_polygon needs at least 4 points" in joined
    assert "regions.jsonl:2: invalid JSON" in joined


def test_series_level_split_when_enough_series() -> None:
    pages = [{"page_id": f"S{s}-P{p}", "series_id": f"S{s}"} for s in range(6) for p in range(3)]
    split = ingest_gold.split_pages(pages)
    for s in range(6):
        assert len({split[f"S{s}-P{p}"] for p in range(3)}) == 1  # a series never spans splits
    assert set(split.values()) == {"dev", "val", "test"}
