"""Create and verify the frozen v0.1.0 baseline worktree (Phase 0.2.1).

Usage: python tools/baseline_worktree.py [--verify]

The worktree (default `.baseline/`, gitignored) is a detached checkout of the commit in
`benchmarks/baseline.lock.json`. The tool refuses to proceed when the tag no longer points at
that commit, or when the worktree is dirty or checked out elsewhere, so every baseline run
executes exactly v0.1.0.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class BaselineError(RuntimeError):
    pass


@dataclass(frozen=True)
class BaselineLock:
    tag: str
    commit: str
    worktree: str


def read_lock(path: Path) -> BaselineLock:
    data = json.loads(path.read_text(encoding="utf-8"))
    return BaselineLock(str(data["tag"]), str(data["commit"]), str(data["worktree"]))


def _git(cwd: Path, *args: str) -> str:
    out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise BaselineError(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout.strip()


def _check_tag(repo: Path, lock: BaselineLock) -> None:
    tagged = _git(repo, "rev-parse", f"{lock.tag}^{{commit}}")
    if tagged != lock.commit:
        raise BaselineError(f"tag {lock.tag} points at {tagged}, expected {lock.commit}")


def verify(repo: Path, lock: BaselineLock) -> Path:
    _check_tag(repo, lock)
    wt = repo / lock.worktree
    if not (wt / ".git").exists():
        raise BaselineError(f"baseline worktree {wt} does not exist")
    head = _git(wt, "rev-parse", "HEAD")
    if head != lock.commit:
        raise BaselineError(f"baseline worktree HEAD is {head}, expected {lock.commit}")
    if _git(wt, "status", "--porcelain", "--untracked-files=no"):
        raise BaselineError(f"baseline worktree {wt} is dirty")
    return wt


def ensure(repo: Path, lock: BaselineLock) -> Path:
    _check_tag(repo, lock)
    wt = repo / lock.worktree
    if not (wt / ".git").exists():
        _git(repo, "worktree", "add", "--detach", str(wt), lock.commit)
    return verify(repo, lock)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verify", action="store_true", help="verify only, never create")
    args = ap.parse_args(argv)
    lock = read_lock(ROOT / "benchmarks" / "baseline.lock.json")
    try:
        wt = verify(ROOT, lock) if args.verify else ensure(ROOT, lock)
    except BaselineError as exc:
        sys.stderr.write(f"baseline: {exc}\n")
        return 1
    sys.stdout.write(f"baseline OK: {wt} at {lock.tag} ({lock.commit[:12]})\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
