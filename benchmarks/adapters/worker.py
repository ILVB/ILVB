"""Adapter worker process: ``python -m benchmarks.adapters.worker IMPL JOBS OUT``.

IMPL is ``baseline`` or ``candidate``; the client sets PYTHONPATH so that ``manga_ar``
resolves to `.baseline/src` or `src`. JOBS is a JSONL file of jobs; OUT a directory that
receives ``<page_id>.<mode>.json`` (a PageResult) plus images. One process handles all
jobs so models load once. Errors are recorded per job, never raised past it.
"""

from __future__ import annotations

import importlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from benchmarks.schema import GtPage, PageResult

IMPLS = {"baseline": "benchmarks.adapters.v010", "candidate": "benchmarks.adapters.candidate"}


def _load(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _save(path: Path, rgb: np.ndarray) -> str:
    Image.fromarray(rgb).save(path)
    return path.name


def run_job(engines: dict[str, Any], impl: str, job: dict[str, Any], out: Path) -> PageResult:
    module = importlib.import_module(IMPLS[impl])
    key = json.dumps(job.get("config", {}), sort_keys=True) + json.dumps(job.get("tm", {}))
    if key not in engines:
        engines[key] = module.Engine(job.get("config", {}), job.get("tm") or None)
    engine = engines[key]
    mode, page_id = job["mode"], job["page_id"]
    result: PageResult
    stem = out / f"{page_id}.{mode}"
    if mode == "detect_ocr":
        result = engine.detect_ocr(_load(job["image"]), page_id, job["lang"])
    elif mode == "erase":
        result, erased = engine.erase(_load(job["image"]), page_id, job["lang"])
        result.erased_image = _save(Path(f"{stem}.erased.png"), erased)
    elif mode == "typeset_gt":
        gt = GtPage.model_validate_json(Path(job["gt"]).read_text(encoding="utf-8"))
        result, final = engine.typeset_gt(
            _load(job["image"]), _load(job["clean"]), gt, job.get("texts", {})
        )
        result.final_image = _save(Path(f"{stem}.final.png"), final)
    elif mode == "translate_gt":
        gt = GtPage.model_validate_json(Path(job["gt"]).read_text(encoding="utf-8"))
        result = engine.translate_gt(gt)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    result.version = impl
    return result


def main(argv: list[str]) -> int:
    impl, jobs_path, out_dir = argv
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    engines: dict[str, Any] = {}
    for line in Path(jobs_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        job = json.loads(line)
        try:
            result = run_job(engines, impl, job, out)
        except Exception as exc:  # noqa: BLE001 - one failing item must not stop the run
            result = PageResult(
                page_id=job["page_id"],
                version=impl,
                mode=job["mode"],
                errors=[f"{type(exc).__name__}: {exc}", traceback.format_exc(limit=5)],
            )
        target = out / f"{job['page_id']}.{job['mode']}.json"
        target.write_text(result.model_dump_json(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
