"""G-LIC-1 check: every dependency introduced since v0.1.0 is in docs/LICENSES.md.

Usage: python -m tools.check_licenses [--repo .]

Compares the packages resolved in uv.lock at HEAD with those at the frozen baseline commit
(benchmarks/baseline.lock.json); each new package must appear in a table of
docs/LICENSES.md. Also fails if the register marks anything non-commercial without saying
it is opt-in or not bundled. Models, fonts and datasets are checked by
tests/unit/test_license_register.py (they need the package importable).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from tools.phase_map import dist_name, load_toml

REGISTER = Path("docs/LICENSES.md")


def lock_names(text: str) -> set[str]:
    return {dist_name(p["name"]) for p in load_toml(text).get("package", [])}


def register_names(text: str) -> set[str]:
    names = set()
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and cells and not set(cells[0]) <= {"-", ""}:
            names.add(dist_name(cells[0]))
    return names


def problems(repo: Path) -> list[str]:
    path = repo / REGISTER
    if not path.is_file():
        return [f"{REGISTER} missing"]
    text = path.read_text("utf-8")
    base = json.loads((repo / "benchmarks" / "baseline.lock.json").read_text("utf-8"))["commit"]
    old = subprocess.run(["git", "-C", str(repo), "show", f"{base}:uv.lock"],
                         capture_output=True, text=True, check=True).stdout  # fmt: skip
    new = lock_names((repo / "uv.lock").read_text("utf-8")) - lock_names(old)
    listed = register_names(text)
    out = [f"{n}: introduced since v0.1.0 but not in {REGISTER}" for n in sorted(new - listed)]
    for line in text.splitlines():
        nc = line.startswith("|") and re.search(r"non-commercial", line, re.I)
        if nc and not re.search(r"opt-in|not bundled|not used", line, re.I):
            out.append(f"non-commercial entry without opt-in/not-bundled status: {line[:80]}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    args = ap.parse_args(argv)
    found = problems(args.repo)
    for p in found:
        sys.stdout.write(f"FAIL {p}\n")
    sys.stdout.write(f"check_licenses: {'FAIL' if found else 'OK'}\n")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
