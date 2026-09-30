"""Assert that Gate 1 passed and is still valid (PD-1 hard gate; Phase 2 step 2.0).

Usage: python -m tools.assert_gate [--repo .] [--if-phase2-staged]

Exits non-zero unless: .gates/phase1.approved exists; its recorded SHA-256 matches
benchmarks/results/phase1_gate.json; that JSON's verdict is PASS; the recorded commit is
an ancestor of HEAD; no Phase-1-scoped file changed since that commit (working tree
included); and the recorded human message contains the exact phrase PHASE 1 APPROVED.

--if-phase2-staged (pre-commit hook) runs the check only when the index touches a
Phase 2 path, and --if-phase2-present (CI) only when a tracked file lies under a Phase 2
path, so Phase 0/1 work is not blocked by the missing gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from tools import phase_map


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          check=False)  # fmt: skip


def problems(repo: Path) -> list[str]:
    approval = repo / phase_map.APPROVAL_FILE
    if not approval.is_file():
        return [f"{phase_map.APPROVAL_FILE} missing: Gate 1 has not been approved"]
    try:
        record = json.loads(approval.read_text("utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{phase_map.APPROVAL_FILE} is not valid JSON ({exc.msg})"]
    out: list[str] = []
    if phase_map.APPROVAL_PHRASE not in str(record.get("human_message", "")):
        out.append(f"recorded human message lacks the exact phrase {phase_map.APPROVAL_PHRASE}")
    gate = repo / phase_map.GATE_JSON
    if not gate.is_file():
        return [*out, f"{phase_map.GATE_JSON} missing"]
    if hashlib.sha256(gate.read_bytes()).hexdigest() != record.get("gate_json_sha256"):
        out.append(f"{phase_map.GATE_JSON} does not match the approved SHA-256")
    try:
        verdict = json.loads(gate.read_text("utf-8")).get("verdict")
    except json.JSONDecodeError as exc:
        verdict = f"invalid JSON ({exc.msg})"
    if verdict != "PASS":
        out.append(f"gate verdict is {verdict!r}, not 'PASS'")
    commit = str(record.get("commit", ""))
    if _git(repo, "merge-base", "--is-ancestor", commit, "HEAD").returncode != 0:
        return [*out, f"approved commit {commit[:10] or '?'} is not an ancestor of HEAD"]
    changed = _git(repo, "diff", "--name-only", commit, "--").stdout.splitlines()
    stale = [f for f in changed if phase_map.is_phase1_scoped(f)]
    if stale:
        out.append(f"{len(stale)} Phase-1-scoped file(s) changed since the approved commit "
                   f"(re-run the gate): {', '.join(stale[:5])}")  # fmt: skip
    return out


def phase2_staged(repo: Path) -> bool:
    staged = _git(repo, "diff", "--cached", "--name-only").stdout.splitlines()
    return any(phase_map.is_phase2_path(f) for f in staged)


def phase2_present(repo: Path) -> bool:
    return any(phase_map.is_phase2_path(f) for f in _git(repo, "ls-files").stdout.splitlines())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--if-phase2-staged", action="store_true")
    mode.add_argument("--if-phase2-present", action="store_true")
    args = ap.parse_args(argv)
    if args.if_phase2_staged and not phase2_staged(args.repo):
        return 0
    if args.if_phase2_present and not phase2_present(args.repo):
        return 0
    found = problems(args.repo)
    for p in found:
        sys.stdout.write(f"FAIL {p}\n")
    sys.stdout.write(f"assert_gate: {'FAIL' if found else 'OK'}\n")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
