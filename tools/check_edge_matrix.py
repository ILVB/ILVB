"""Check docs/EDGE_CASE_MATRIX.md against the test suite (edge cases E-01..E-18).

Usage: python -m tools.check_edge_matrix [--root .]

Every ID must have at least one test function whose name contains it (``test_e05_...``),
every test the matrix names must exist, and each row's status must match reality
(``covered`` exactly when a test exists). Exit 1 on any mismatch or untested ID.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

IDS = tuple(f"E-{n:02d}" for n in range(1, 19))
MATRIX = Path("docs/EDGE_CASE_MATRIX.md")
_TEST = re.compile(r"^\s*(?:async\s+)?def\s+(test\w*)\s*\(", re.MULTILINE)
_ROW = re.compile(r"^\|\s*(E-\d{2})\s*\|(.*)$", re.MULTILINE)


def id_of(test_name: str) -> list[str]:
    return [f"E-{m}" for m in re.findall(r"(?:^|_)e(\d{2})(?=_|$)", test_name.lower())]


def collect_tests(root: Path) -> dict[str, list[str]]:
    """ID → ["tests/path.py::test_name", ...] for every ID-named test function."""
    found: dict[str, list[str]] = defaultdict(list)
    for path in sorted((root / "tests").rglob("test_*.py")):
        for name in _TEST.findall(path.read_text("utf-8")):
            for eid in id_of(name):
                found[eid].append(f"{path.relative_to(root).as_posix()}::{name}")
    return found


def parse_matrix(text: str) -> dict[str, dict[str, str]]:
    rows = {}
    for eid, rest in _ROW.findall(text):
        cells = [c.strip() for c in rest.strip().strip("|").split("|")]
        rows[eid] = {"tests": cells[-2] if len(cells) >= 2 else "", "status": cells[-1]}
    return rows


def problems(root: Path) -> list[str]:
    path = root / MATRIX
    if not path.is_file():
        return [f"{MATRIX} missing"]
    rows, tests = parse_matrix(path.read_text("utf-8")), collect_tests(root)
    out = [f"{eid}: no row in the matrix" for eid in IDS if eid not in rows]
    for eid in IDS:
        if eid not in rows:
            continue
        listed = re.findall(r"`([^`]+::test\w*)`", rows[eid]["tests"])
        out += [f"{eid}: listed test {t} does not exist" for t in listed if t not in tests[eid]]
        out += [f"{eid}: test {t} is not listed" for t in tests[eid] if t not in listed]
        covered = rows[eid]["status"].lower().startswith("covered")
        if not tests[eid]:
            out.append(f"{eid}: untested (every ID needs a test whose name contains it)")
        if covered != bool(tests[eid]):
            out.append(f"{eid}: status {rows[eid]['status']!r} does not match the tests")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=Path.cwd())
    args = ap.parse_args(argv)
    found = problems(args.root)
    for p in found:
        sys.stdout.write(f"FAIL {p}\n")
    tested = sum(1 for eid, t in collect_tests(args.root).items() if eid in IDS and t)
    sys.stdout.write(f"edge matrix: {tested}/{len(IDS)} IDs tested; "
                     f"{'FAIL' if found else 'OK'}\n")  # fmt: skip
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
