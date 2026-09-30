"""Validate and ingest the human gold set (step 0.3.5).

Usage: python tools/ingest_gold.py [--src benchmarks/data/gold_real] [--check]

Reads pages.csv, regions.jsonl, arabic_references.csv, glossary.yaml and speakers.csv
(formats: benchmarks/data/templates/README.md). --check validates and reports problems
with file:line locations. Without --check it writes GtPage JSON files into
`<src>/ingested/<split>/`, splitting by series (by page when fewer than 3 series), and
records only per-split content hashes in the committed `benchmarks/manifests/gold_real.sha256`
(real content never enters git).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TYPES = {"dialogue", "thought", "narration", "sfx", "sign", "credit"}


@dataclass
class Report:
    problems: list[str] = field(default_factory=list)

    def add(self, where: str, message: str) -> None:
        self.problems.append(f"{where}: {message}")


def _csv(path: Path, required: set[str], report: Report) -> list[dict[str, str]]:
    if not path.is_file():
        report.add(str(path), "missing")
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = required - set(reader.fieldnames or [])
        if missing:
            report.add(str(path), f"missing columns {sorted(missing)}")
            return []
        return [dict(row) for row in reader]


def validate(src: Path) -> tuple[Report, dict[str, Any]]:
    report = Report()
    pages = _csv(src / "pages.csv", {"page_id", "series_id", "file_name", "source_lang",
                                     "reading_direction"}, report)  # fmt: skip
    page_ids = {p["page_id"] for p in pages}
    for i, p in enumerate(pages, start=2):
        if p["source_lang"] not in {"ja", "zh", "ko", "en"}:
            report.add(f"pages.csv:{i}", f"source_lang {p['source_lang']!r} not in ja/zh/ko/en")
        if p["reading_direction"] not in {"rtl", "ltr"}:
            report.add(f"pages.csv:{i}", "reading_direction must be rtl or ltr")
        if not (src / "images" / p["file_name"]).is_file():
            report.add(f"pages.csv:{i}", f"image images/{p['file_name']} not found")
    regions: list[dict[str, Any]] = []
    rpath = src / "regions.jsonl"
    if not rpath.is_file():
        report.add(str(rpath), "missing")
    else:
        for n, line in enumerate(rpath.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as exc:
                report.add(f"regions.jsonl:{n}", f"invalid JSON ({exc.msg})")
                continue
            for key in ("page_id", "region_id", "type", "text_polygon", "text", "reading_order"):
                if key not in r:
                    report.add(f"regions.jsonl:{n}", f"missing {key!r}")
            if r.get("type") not in TYPES:
                report.add(f"regions.jsonl:{n}", f"type {r.get('type')!r} not in {sorted(TYPES)}")
            if r.get("page_id") not in page_ids:
                report.add(f"regions.jsonl:{n}", f"unknown page_id {r.get('page_id')!r}")
            if len(r.get("text_polygon") or []) < 4:
                report.add(f"regions.jsonl:{n}", "text_polygon needs at least 4 points")
            regions.append(r)
    refs = _csv(src / "arabic_references.csv", {"region_id", "arabic_reference", "post_edited_mt"},
                report)  # fmt: skip
    region_ids = {r.get("region_id") for r in regions}
    for i, row in enumerate(refs, start=2):
        if row["region_id"] not in region_ids:
            report.add(f"arabic_references.csv:{i}", f"unknown region_id {row['region_id']!r}")
    glossary: dict[str, Any] = {}
    gpath = src / "glossary.yaml"
    if gpath.is_file():
        glossary = yaml.safe_load(gpath.read_text(encoding="utf-8")) or {}
    speakers = _csv(src / "speakers.csv", {"series_id", "character_id", "gender"}, report)
    data = {"pages": pages, "regions": regions, "refs": refs, "glossary": glossary,
            "speakers": speakers}  # fmt: skip
    return report, data


def _poly_mask(poly: list[list[float]], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.round(np.array(poly)).astype(np.int32)], 1)
    return mask > 0


def split_pages(pages: list[dict[str, str]]) -> dict[str, str]:
    """Series-level split when there are >= 3 series, else page-level (by keyed hash)."""
    series = sorted({p["series_id"] for p in pages})
    unit = "series_id" if len(series) >= 3 else "page_id"
    keys = sorted(
        {p[unit] for p in pages}, key=lambda k: hashlib.sha256(f"gold:{k}".encode()).hexdigest()
    )
    names = ("dev", "val", "test")
    assign = {k: names[min(2, i * 3 // max(1, len(keys)))] for i, k in enumerate(keys)}
    return {p["page_id"]: assign[p[unit]] for p in pages}


def ingest(src: Path, data: dict[str, Any]) -> dict[str, str]:
    from benchmarks import rle
    from benchmarks.schema import GtPage, GtRegion

    splits = split_pages(data["pages"])
    refs: dict[str, list[str]] = {}
    for row in data["refs"]:
        if row["post_edited_mt"].strip().lower() != "yes":
            refs.setdefault(row["region_id"], []).append(row["arabic_reference"])
    digests: dict[str, Any] = {s: hashlib.sha256() for s in ("dev", "val", "test")}
    for p in sorted(data["pages"], key=lambda x: x["page_id"]):
        img = Image.open(src / "images" / p["file_name"])
        w, h = img.size
        regions = []
        for r in sorted((r for r in data["regions"] if r["page_id"] == p["page_id"]),
                        key=lambda r: r["reading_order"]):  # fmt: skip
            text = _poly_mask(r["text_polygon"], h, w)
            bubble = _poly_mask(r["bubble_polygon"], h, w) if r.get("bubble_polygon") else None
            if bubble is not None:
                ys, xs = np.nonzero(bubble)
                m = max(3, round(0.06 * min(np.ptp(xs) + 1, np.ptp(ys) + 1)))
                safe = cv2.erode(bubble.astype(np.uint8), np.ones((2 * m + 1,) * 2, np.uint8)) > 0
            else:
                safe = cv2.dilate(text.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            text_rle, safe_rle = rle.encode(text), rle.encode(safe)
            if text_rle is None or safe_rle is None:
                raise ValueError(f"{r['region_id']}: empty text polygon")
            regions.append(GtRegion(
                region_id=r["region_id"], category="real", type=r["type"], lang=p["source_lang"],
                text="" if r.get("illegible") else r["text"], vertical=bool(r.get("vertical")),
                text_polygon=[tuple(pt) for pt in r["text_polygon"]], text_mask=text_rle,
                bubble_mask=rle.encode(bubble) if bubble is not None else None,
                safe_mask=safe_rle, reading_order=int(r["reading_order"]),
                speaker=r.get("speaker"), references_ar=refs.get(r["region_id"], []),
                reference_kind="gold" if r["region_id"] in refs else "none",
            ))  # fmt: skip
        gt = GtPage.model_validate({
            "page_id": p["page_id"], "series_id": p["series_id"], "split": splits[p["page_id"]],
            "categories": ["real"], "lang": p["source_lang"],
            "reading_direction": p["reading_direction"], "width": w, "height": h,
            "image": str(src / "images" / p["file_name"]), "clean": "", "seed": 0,
            "generator": "gold_real", "regions": regions,
        })  # fmt: skip
        blob = gt.model_dump_json().encode("utf-8")
        out = src / "ingested" / gt.split / f"{gt.page_id}.gt.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(blob)
        digests[gt.split].update(hashlib.sha256(blob).digest())
    return {s: d.hexdigest() for s, d in digests.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=ROOT / "benchmarks" / "data" / "gold_real")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    report, data = validate(args.src)
    for problem in report.problems:
        sys.stdout.write(problem + "\n")
    if report.problems:
        return 1
    if args.check:
        sys.stdout.write(f"OK: {len(data['pages'])} pages, {len(data['regions'])} regions, "
                         f"{len(data['refs'])} references\n")  # fmt: skip
        return 0
    digests = ingest(args.src, data)
    target = ROOT / "benchmarks" / "manifests" / "gold_real.sha256"
    target.write_text(
        "".join(f"{v}  gold_real/{k}\n" for k, v in digests.items()), encoding="utf-8"
    )
    sys.stdout.write(f"ingested; split hashes written to {target}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
