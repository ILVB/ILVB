"""0.5: assert_gate on throwaway repos, with negative controls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tests.tools.test_audit_phase_order import OK_TRAILERS, _git, commit, repo
from tools import assert_gate, phase_map

__all__ = ["repo"]  # the fixture is shared with the audit tests


def _approve(repo: Path, verdict: str = "PASS", phrase: str = "PHASE 1 APPROVED") -> None:
    gate = json.dumps({"verdict": verdict})
    sha = commit(repo, {phase_map.GATE_JSON: gate}, f"chore: gate\n\n{OK_TRAILERS}")
    record = {"gate_json_sha256": hashlib.sha256(gate.encode()).hexdigest(), "commit": sha,
              "utc": "2026-10-01T00:00:00+00:00", "human_message": phrase}  # fmt: skip
    commit(repo, {phase_map.APPROVAL_FILE: json.dumps(record)}, f"chore: ok\n\n{OK_TRAILERS}")


def test_assert_gate_requires_approval(repo: Path) -> None:
    assert any("missing" in p for p in assert_gate.problems(repo))
    assert assert_gate.main(["--repo", str(repo), "--if-phase2-present"]) == 0  # Phase 0/1
    assert assert_gate.main(["--repo", str(repo), "--if-phase2-staged"]) == 0
    commit(repo, {"src/manga_ar/reader/app.py": "x = 1\n"}, f"feat\n\n{OK_TRAILERS}")
    assert assert_gate.main(["--repo", str(repo), "--if-phase2-present"]) == 1


def test_assert_gate_passes_and_stays_valid_for_docs(repo: Path) -> None:
    _approve(repo)
    assert assert_gate.problems(repo) == []
    commit(repo, {"docs/notes.md": "x", "src/manga_ar/edu/a.py": "x"}, f"docs\n\n{OK_TRAILERS}")
    assert assert_gate.problems(repo) == []


@pytest.mark.parametrize(
    ("verdict", "phrase", "needle"),
    [("FAIL", "PHASE 1 APPROVED", "verdict"), ("PASS", "looks good", "exact phrase")],
)
def test_assert_gate_negative_controls(repo: Path, verdict: str, phrase: str, needle: str) -> None:
    _approve(repo, verdict, phrase)
    assert any(needle in p for p in assert_gate.problems(repo))


def test_assert_gate_invalidated_by_tampering_and_phase1_changes(repo: Path) -> None:
    _approve(repo)
    (repo / phase_map.GATE_JSON).write_text('{"verdict": "PASS", "x": 1}', encoding="utf-8")
    assert any("SHA-256" in p for p in assert_gate.problems(repo))
    _git(repo, "checkout", "--", phase_map.GATE_JSON)
    commit(repo, {"src/manga_ar/ocr/x.py": "x = 2\n"}, f"fix\n\n{OK_TRAILERS}")
    assert any("re-run the gate" in p for p in assert_gate.problems(repo))


def test_assert_gate_needs_ancestor_commit(repo: Path) -> None:
    _approve(repo)
    record = json.loads((repo / phase_map.APPROVAL_FILE).read_text("utf-8"))
    record["commit"] = "0" * 40
    (repo / phase_map.APPROVAL_FILE).write_text(json.dumps(record), encoding="utf-8")
    assert any("not an ancestor" in p for p in assert_gate.problems(repo))
