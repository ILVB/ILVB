"""Audit the release history for phase-order violations (PD-1, PD-4, OP-4).

Usage: python -m tools.audit_phase_order [--repo .] [--commit-msg FILE] [--staged]

Checks every commit after the frozen baseline commit (benchmarks/baseline.lock.json) and
fails (exit 1) if, before tag `phase1-gate-passed`: a commit carries `Phase: 2`; any
Phase 2 path exists in a commit tree; a Phase 2 dependency is declared in pyproject.toml
or resolved in uv.lock. It also fails when any commit lacks valid Phase/Step git trailers,
or when a frozen artefact (phase_map.IMMUTABLE) is touched by more than one commit.
Commits listed in tools/trailer_exceptions.json (Phase/Step lines present in the body but
outside the trailer block) are verified line by line and reported as WARN on every run.

--commit-msg FILE validates a message before it is committed (commit-msg hook); --staged
checks the index for Phase 2 paths and dependencies (pre-commit hook).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from tools import phase_map

EXCEPTIONS = Path("tools/trailer_exceptions.json")
_PHASES = {"0", "1", "2"}


@dataclass(frozen=True)
class Finding:
    commit: str
    message: str
    warn: bool = False

    def __str__(self) -> str:
        return f"{'WARN' if self.warn else 'FAIL'} {self.commit[:10]}: {self.message}"


def git(repo: Path, *args: str, check: bool = True) -> str:
    out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                         check=check)  # fmt: skip
    return out.stdout


def trailers(message: str) -> dict[str, list[str]]:
    """Git trailers of ``message`` (last paragraph only, exactly as git parses them)."""
    out = subprocess.run(["git", "interpret-trailers", "--parse"], input=message,
                         capture_output=True, text=True, check=True).stdout  # fmt: skip
    found: dict[str, list[str]] = {}
    for line in out.splitlines():
        key, _, value = line.partition(":")
        found.setdefault(key.strip().lower(), []).append(value.strip())
    return found


def trailer_problem(phase: list[str], step: list[str]) -> str | None:
    if len(phase) != 1 or len(step) != 1:
        return "needs exactly one Phase and one Step trailer"
    if phase[0] not in _PHASES:
        return f"invalid Phase {phase[0]!r}"
    if not re.fullmatch(rf"{phase[0]}(\.\d+)*", step[0]):
        return f"Step {step[0]!r} does not belong to Phase {phase[0]}"
    return None


def body_lines(message: str, key: str) -> list[str]:
    return [m.group(1).strip() for m in re.finditer(rf"(?mi)^{key}:\s*(\S.*)$", message)]


def base_commit(repo: Path) -> str:
    lock = json.loads((repo / "benchmarks" / "baseline.lock.json").read_text("utf-8"))
    return str(lock["commit"])


def gate_commit(repo: Path) -> str | None:
    out = subprocess.run(["git", "-C", str(repo), "rev-parse", "-q", "--verify",
                          f"refs/tags/{phase_map.GATE_TAG}^{{commit}}"],
                         capture_output=True, text=True, check=False)  # fmt: skip
    return out.stdout.strip() or None


def _after_gate(repo: Path, sha: str, gate: str | None) -> bool:
    if gate is None or sha == gate:
        return False
    ok = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", gate, sha],
                        capture_output=True, check=False)  # fmt: skip
    return ok.returncode == 0


def dependency_problems(pyproject_text: str | None, lock_text: str | None) -> list[str]:
    out = []
    if pyproject_text:
        out += phase_map.pyproject_violations(phase_map.load_toml(pyproject_text))
    if lock_text:
        out += phase_map.lock_violations(phase_map.load_toml(lock_text))
    return out


def _show(repo: Path, sha: str, path: str) -> str | None:
    out = subprocess.run(["git", "-C", str(repo), "show", f"{sha}:{path}"],
                         capture_output=True, text=True, check=False)  # fmt: skip
    return out.stdout if out.returncode == 0 else None


def _exceptions(repo: Path) -> dict[str, dict[str, str]]:
    path = repo / EXCEPTIONS
    if not path.is_file():
        return {}
    return {e["commit"]: e for e in json.loads(path.read_text("utf-8"))["commits"]}


def check_trailers(sha: str, message: str, exceptions: dict[str, dict[str, str]]) -> list[Finding]:
    t = trailers(message)
    problem = trailer_problem(t.get("phase", []), t.get("step", []))
    if problem is None:
        return []
    exc = exceptions.get(sha)
    if exc is None:
        return [Finding(sha, problem)]
    phase, step = body_lines(message, "Phase"), body_lines(message, "Step")
    if phase != [exc["phase"]] or step != [exc["step"]] or trailer_problem(phase, step):
        return [Finding(sha, f"listed exception does not match the message ({problem})")]
    return [Finding(sha, f"Phase/Step outside the trailer block (acknowledged: "
                         f"Phase {exc['phase']}, Step {exc['step']})", warn=True)]  # fmt: skip


def audit(repo: Path) -> list[Finding]:
    base, gate = base_commit(repo), gate_commit(repo)
    exceptions = _exceptions(repo)
    findings: list[Finding] = []
    for sha in git(repo, "rev-list", "--reverse", f"{base}..HEAD").split():
        message = git(repo, "log", "-1", "--format=%B", sha)
        findings += check_trailers(sha, message, exceptions)
        if _after_gate(repo, sha, gate):
            continue
        if trailers(message).get("phase") == ["2"]:
            findings.append(Finding(sha, f"Phase 2 commit before tag {phase_map.GATE_TAG}"))
        files = git(repo, "ls-tree", "-r", "--name-only", sha).splitlines()
        findings += [Finding(sha, f"Phase 2 path before the gate: {f}")
                     for f in files if phase_map.is_phase2_path(f)]  # fmt: skip
        deps = dependency_problems(_show(repo, sha, "pyproject.toml"), _show(repo, sha, "uv.lock"))
        findings += [Finding(sha, f"Phase 2 dependency before the gate: {d}") for d in deps]
    for path in phase_map.IMMUTABLE:
        touching = git(repo, "log", "--format=%H", f"{base}..HEAD", "--", path).split()
        if len(touching) > 1:
            what = f"frozen {path} changed after its creation ({len(touching)} commits touch it)"
            findings.append(Finding(touching[0], what))
    return findings


def check_staged(repo: Path) -> list[Finding]:
    if gate_commit(repo) is not None:
        return []
    files = git(repo, "diff", "--cached", "--name-only", "--diff-filter=ACMR").splitlines()
    out = [Finding("index", f"Phase 2 path before the gate: {f}")
           for f in files if phase_map.is_phase2_path(f)]  # fmt: skip
    staged = {p: _show(repo, "", p) for p in ("pyproject.toml", "uv.lock")}  # ":path" = index
    out += [Finding("index", f"Phase 2 dependency before the gate: {d}")
            for d in dependency_problems(staged["pyproject.toml"], staged["uv.lock"])]  # fmt: skip
    return out


def check_message(repo: Path, message: str) -> list[Finding]:
    t = trailers(message)
    problem = trailer_problem(t.get("phase", []), t.get("step", []))
    out = [Finding("message", problem)] if problem else []
    if t.get("phase") == ["2"] and gate_commit(repo) is None:
        out.append(Finding("message", f"Phase 2 commit before tag {phase_map.GATE_TAG}"))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--commit-msg", type=Path, default=None)
    ap.add_argument("--staged", action="store_true")
    args = ap.parse_args(argv)
    if args.commit_msg is not None:
        text = args.commit_msg.read_text("utf-8")
        findings = check_message(args.repo, re.sub(r"(?m)^#.*\n?", "", text))
    elif args.staged:
        findings = check_staged(args.repo)
    else:
        findings = audit(args.repo)
    for f in findings:
        sys.stdout.write(f"{f}\n")
    failed = [f for f in findings if not f.warn]
    warned = len(findings) - len(failed)
    verdict = "FAIL" if failed else "OK"
    sys.stdout.write(f"audit_phase_order: {verdict} ({len(failed)} failures, {warned} warnings)\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
