"""Run adapter jobs in a worker subprocess with the right code on ``sys.path``."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from benchmarks.schema import PageResult

ROOT = Path(__file__).resolve().parents[2]


def code_dir(impl: str, root: Path = ROOT) -> Path:
    if impl == "baseline":
        from tools.baseline_worktree import read_lock, verify

        lock = read_lock(root / "benchmarks" / "baseline.lock.json")
        return verify(root, lock) / "src"
    if impl == "candidate":
        return root / "src"
    raise ValueError(f"unknown implementation {impl!r}")


def worker_env(impl: str, root: Path = ROOT) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(code_dir(impl, root)), str(root)])
    env.setdefault("MANGAAR_CACHE_DIR", str(root / ".cache"))
    env["PYTHONHASHSEED"] = "0"
    env["HF_HUB_OFFLINE"] = "1"
    return env


def run_jobs(
    impl: str, jobs: list[dict[str, Any]], out: Path, root: Path = ROOT
) -> list[PageResult]:
    out.mkdir(parents=True, exist_ok=True)
    jobs_path = out / f"jobs.{impl}.jsonl"
    jobs_path.write_text("".join(json.dumps(j) + "\n" for j in jobs), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "benchmarks.adapters.worker", impl, str(jobs_path), str(out)],
        env=worker_env(impl, root),
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{impl} worker failed ({proc.returncode}): {proc.stderr[-4000:]}")
    return [
        PageResult.model_validate_json(
            (out / f"{j['page_id']}.{j['mode']}.json").read_text(encoding="utf-8")
        )
        for j in jobs
    ]
