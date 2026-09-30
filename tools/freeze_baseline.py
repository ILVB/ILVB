"""Freeze the v0.1.0 baseline results for every split, once (step 0.2.4).

Usage: python -m tools.freeze_baseline

Runs the baseline adapter (the `.baseline/` worktree at the locked tag) over dev, val and
the sealed test split, and writes `benchmarks/results/baseline_v0.1.0.json` plus its
`.sha256`. The file is immutable: the tool refuses to overwrite it, and the gate
re-verifies the hash. Test-split results are written, never printed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks import datasets, run_benchmark

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "benchmarks" / "results" / "baseline_v0.1.0.json"
DATASET = "synthetic_v1"
PROFILE = "v010-offline"


def freeze(target: Path = FROZEN) -> dict[str, Any]:
    if target.exists():
        raise FileExistsError(f"{target} is frozen; it is never regenerated (PD-3)")
    splits: dict[str, Any] = {}
    for split in ("dev", "val", "test"):
        cap = datasets._issue("freeze-baseline") if split == "test" else None
        splits[split] = run_benchmark.run("baseline", split, capability=cap, dataset=DATASET,
                                          profile=PROFILE)  # fmt: skip
    manifest = datasets.MANIFEST_DIR / f"{DATASET}.json"
    frozen = {
        "schema": 1, "frozen_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": DATASET, "manifest_sha256": datasets.sha256(manifest),
        "profile": PROFILE, "splits": splits,
    }  # fmt: skip
    blob = (json.dumps(frozen, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(blob)
    digest = hashlib.sha256(blob).hexdigest()
    target.with_name(target.name + ".sha256").write_text(f"{digest}  {target.name}\n", "utf-8")
    return frozen


def verify(target: Path = FROZEN) -> bool:
    """True when the frozen file still matches its recorded SHA-256."""
    recorded = target.with_name(target.name + ".sha256").read_text("utf-8").split()[0]
    return hashlib.sha256(target.read_bytes()).hexdigest() == recorded


def main(argv: list[str] | None = None) -> int:
    del argv
    try:
        frozen = freeze()
    except FileExistsError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    for split in ("dev", "val"):
        s = frozen["splits"][split]["summary"]
        sys.stdout.write(f"{split}: {json.dumps(s, ensure_ascii=False)}\n")
    pages = frozen["splits"]["test"]["summary"]["pages"]
    sys.stdout.write(f"test: sealed ({pages} pages scored; results not printed)\n")
    sys.stdout.write(f"frozen: {FROZEN.relative_to(ROOT)} (sha256 recorded)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
