"""Freeze the v0.1.0 *online* translation baseline once (ADR-0007, item 3).

Usage (on a host that can reach translate.google.com):
    python -m tools.freeze_online_baseline

Runs the locked v0.1.0 worktree in `translate_gt` mode with its Google provider (via
deep-translator) and the translation cache on, over dev, val and the sealed test split.
It writes `benchmarks/results/baseline_v0.1.0_online.json` and its `.sha256`. Only synthetic
benchmark sentences leave the machine, as the human authorised. The file is immutable once
committed, and test results are written, never printed. The offline baseline
(`baseline_v0.1.0.json`) is unchanged; this file is the `v0.1.0_online_baseline` that G-TR
compares against.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks import datasets, run_benchmark

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "benchmarks" / "results" / "baseline_v0.1.0_online.json"
DATASET = "synthetic_v1"
PROFILE = "v010-online"
PROBE = "https://translate.google.com/"


def reachable(url: str = PROBE, timeout: float = 8.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return bool(200 <= resp.status < 400)
    except OSError:
        return False


def freeze(target: Path = FROZEN, online: bool | None = None) -> dict[str, Any]:
    if target.exists():
        raise FileExistsError(f"{target} is frozen; it is never regenerated (PD-3)")
    if not (reachable() if online is None else online):
        raise ConnectionError(f"{PROBE} is unreachable from this host; run this on a host "
                              "that can reach it (ADR-0007 item 3)")  # fmt: skip
    splits: dict[str, Any] = {}
    for split in ("dev", "val", "test"):
        cap = datasets._issue("freeze-online-baseline") if split == "test" else None
        splits[split] = run_benchmark.run("baseline", split, capability=cap, dataset=DATASET,
                                          profile=PROFILE, modes=("translate_gt",),
                                          tag="online")  # fmt: skip
    frozen = {
        "schema": 1, "name": "v0.1.0_online_baseline",
        "frozen_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": DATASET,
        "manifest_sha256": datasets.sha256(datasets.MANIFEST_DIR / f"{DATASET}.json"),
        "profile": PROFILE, "config": run_benchmark.PROFILES[PROFILE], "splits": splits,
    }  # fmt: skip
    blob = (json.dumps(frozen, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(blob)
    digest = hashlib.sha256(blob).hexdigest()
    target.with_name(target.name + ".sha256").write_text(f"{digest}  {target.name}\n", "utf-8")
    return frozen


def main(argv: list[str] | None = None) -> int:
    del argv
    sys.stdout.write("note: synthetic benchmark sentences are sent to Google Translate\n")
    try:
        frozen = freeze()
    except (FileExistsError, ConnectionError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    for split in ("dev", "val"):
        tr = frozen["splits"][split]["summary"].get("translation", {})
        sys.stdout.write(f"{split}: {json.dumps(tr, ensure_ascii=False)}\n")
    sys.stdout.write("test: sealed (results not printed)\n")
    sys.stdout.write(f"frozen: {FROZEN.relative_to(ROOT)}; commit it with its .sha256\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
