"""Run one implementation over one split and score it.

    python -m benchmarks.run_benchmark --version {baseline,candidate} --split {dev,val}

Raw adapter outputs go to `benchmarks/cache/results/<version>/<dataset>/<split>/`
(gitignored). The scored result holds the summary, per-page metric rows and a SHA-256 over
the raw files. The test split is sealed: this CLI refuses it, and only the gate and the
baseline freeze open it through ``run(..., capability=...)``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmarks.adapters.client import run_jobs
from benchmarks.datasets import DATA_DIR, ROOT, SealedCapability, load_split
from benchmarks.evaluate import load_rgb, score_page, typeset_texts
from benchmarks.schema import PageResult
from benchmarks.summary import summarize

RAW_ROOT = DATA_DIR / "results"
MODES = ("erase", "typeset_gt", "translate_gt")
# Offline (PD-8) and without the persistent translation cache, so results never depend on
# earlier runs. Candidate profiles are added with the Phase 1 feature flags.
PROFILES: dict[str, dict[str, Any]] = {
    "v010-offline": {"runtime.offline": True, "translate.cache": False},
}


def code_revision(version: str) -> dict[str, Any]:
    if version == "baseline":
        from tools.baseline_worktree import read_lock

        lock = read_lock(ROOT / "benchmarks" / "baseline.lock.json")
        return {"tag": lock.tag, "commit": lock.commit, "dirty": False}
    git = ["git", "-C", str(ROOT)]
    head = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    status = subprocess.run([*git, "status", "--porcelain", "--", "src"], capture_output=True,
                            text=True, check=True)  # fmt: skip
    return {"commit": head.stdout.strip(), "dirty": bool(status.stdout.strip())}


VOLATILE = {"stages", "code_origin"}  # timings and absolute paths differ between re-runs


def raw_digest(raw: Path, results: list[PageResult]) -> str:
    """Reproducible SHA-256 over raw outputs: canonical JSON without volatile fields, plus
    decoded pixels (independent of PNG encoder settings). Equal digests = identical runs."""
    digest = hashlib.sha256()
    for res in sorted(results, key=lambda r: (r.page_id, r.mode)):
        body = res.model_dump(mode="json", exclude=VOLATILE)
        digest.update(json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        for name in (res.erased_image, res.final_image):
            if name:
                pixels = load_rgb(raw / name)
                digest.update(f"{name}:{pixels.shape}".encode())
                digest.update(pixels.tobytes())
    return digest.hexdigest()


def run(
    version: str, split: str, *, capability: SealedCapability | None = None,
    dataset: str = "synthetic_v1", profile: str = "v010-offline", limit: int | None = None,
    raw_root: Path = RAW_ROOT,
) -> dict[str, Any]:  # fmt: skip
    items = load_split(split, dataset, capability)[:limit]
    config = PROFILES[profile]
    gts = {it.page_id: it.gt() for it in items}
    jobs = [{"mode": mode, "page_id": it.page_id, "lang": it.lang, "image": str(it.image),
             "clean": str(it.clean), "gt": str(it.gt_path), "config": config,
             "texts": typeset_texts(gts[it.page_id]) if mode == "typeset_gt" else {}}
            for it in items for mode in MODES]  # fmt: skip
    raw = raw_root / version / dataset / split
    results = run_jobs(version, jobs, raw)
    by_page: dict[str, dict[str, PageResult]] = defaultdict(dict)
    for res in results:
        by_page[res.page_id][res.mode] = res
    rows = [score_page(gts[it.page_id], it.category, by_page[it.page_id], raw,
                       load_rgb(it.image), load_rgb(it.clean)) for it in items]  # fmt: skip
    refs = {(g.page_id, r.region_id): r.references_ar for g in gts.values() for r in g.regions}
    kinds = {r.reference_kind for g in gts.values() for r in g.regions if r.references_ar}
    kind = kinds.pop() if len(kinds) == 1 else "mixed"
    origin = Path(results[0].code_origin) if results and results[0].code_origin else None
    return {
        "schema": 1, "version": version, "dataset": dataset, "split": split,
        "profile": profile, "config": config, "limit": limit, "code": code_revision(version),
        "code_origin": str(origin.relative_to(ROOT)) if origin and origin.is_relative_to(ROOT)
        else str(origin),
        "raw_sha256": raw_digest(raw, results),
        "summary": summarize(rows, refs, kind), "items": rows,
    }  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--version", required=True, choices=("baseline", "candidate"))
    ap.add_argument("--split", required=True, choices=("dev", "val", "test"))
    ap.add_argument("--dataset", default="synthetic_v1")
    ap.add_argument("--profile", default="v010-offline", choices=sorted(PROFILES))
    ap.add_argument("--limit", type=int, default=None, help="first N pages (smoke runs)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.split == "test":
        sys.stderr.write(
            "the test split is sealed; only tools/validate_phase1.py evaluates it "
            "(and tools/freeze_baseline.py, once)\n"
        )
        return 2
    result = run(args.version, args.split, dataset=args.dataset, profile=args.profile,
                 limit=args.limit)  # fmt: skip
    out = args.out or RAW_ROOT / f"{args.version}.{args.dataset}.{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps(result["summary"], ensure_ascii=False, indent=1) + "\n")
    sys.stdout.write(f"written: {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
