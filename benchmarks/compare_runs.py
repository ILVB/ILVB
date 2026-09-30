"""Compare two raw runs file by file: ``python -m benchmarks.compare_runs A B --split dev``.

A and B are implementations (``baseline``/``candidate``) whose raw outputs already exist
under benchmarks/cache/results/<impl>/<dataset>/<split>/. Exit 0 when every (page, mode)
output is identical apart from volatile fields (timings, paths, implementation name).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from benchmarks.datasets import load_split
from benchmarks.run_benchmark import MODES, RAW_ROOT, content_digests
from benchmarks.schema import PageResult


def digests(impl: str, split: str, dataset: str, root: Path = RAW_ROOT) -> dict[str, str]:
    raw = root / impl / dataset / split
    results = [
        PageResult.model_validate_json((raw / f"{it.page_id}.{m}.json").read_text("utf-8"))
        for it in load_split(split, dataset, verify=False)
        for m in MODES
    ]
    return content_digests(raw, results)


def differences(a: dict[str, str], b: dict[str, str]) -> list[str]:
    return sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--split", choices=("dev", "val"), default="dev")
    ap.add_argument("--dataset", default="synthetic_v1")
    args = ap.parse_args(argv)
    diff = differences(digests(args.a, args.split, args.dataset),
                       digests(args.b, args.split, args.dataset))  # fmt: skip
    for key in diff:
        sys.stdout.write(f"differs: {key}\n")
    sys.stdout.write(f"{'IDENTICAL' if not diff else f'{len(diff)} outputs differ'}\n")
    return 0 if not diff else 1


if __name__ == "__main__":
    raise SystemExit(main())
