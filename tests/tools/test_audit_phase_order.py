"""0.5: audit_phase_order on throwaway repos, with negative controls."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools import audit_phase_order, phase_map

OK_TRAILERS = "Phase: 0\nStep: 0.5\nCo-Authored-By: A <a@example.com>"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          check=True).stdout.strip()  # fmt: skip


def commit(repo: Path, files: dict[str, str], message: str) -> str:
    for rel, text in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "--allow-empty", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "T")
    base = commit(tmp_path, {"README.md": "v0.1.0\n"}, "release v0.1.0")
    lock = json.dumps({"tag": "v0.1.0", "commit": base, "worktree": ".baseline"})
    commit(tmp_path, {"benchmarks/baseline.lock.json": lock}, f"chore: lock\n\n{OK_TRAILERS}")
    return tmp_path


def fails(repo: Path) -> list[str]:
    return [f.message for f in audit_phase_order.audit(repo) if not f.warn]


def test_clean_history_passes(repo: Path) -> None:
    commit(repo, {"src/manga_ar/ocr/x.py": "x = 1\n"}, f"feat: x\n\n{OK_TRAILERS}")
    assert audit_phase_order.audit(repo) == []


def test_fabricated_phase2_commit_fails(repo: Path) -> None:
    commit(repo, {"notes.txt": "x"}, "feat: reader\n\nPhase: 2\nStep: 2.1")
    assert any("Phase 2 commit before tag" in m for m in fails(repo))


def test_phase2_path_fails_and_passes_after_gate_tag(repo: Path) -> None:
    commit(repo, {"src/manga_ar/edu/card.py": "x = 1\n"}, f"feat: card\n\n{OK_TRAILERS}")
    assert any("src/manga_ar/edu/card.py" in m for m in fails(repo))
    repo2 = repo.parent / "second"
    repo2.mkdir()
    _git(repo2, "init", "-q", "-b", "main")
    _git(repo2, "config", "user.email", "t@example.com")
    _git(repo2, "config", "user.name", "T")
    base = commit(repo2, {"README.md": "v0.1.0\n"}, "release")
    lock = json.dumps({"tag": "v0.1.0", "commit": base, "worktree": ".baseline"})
    commit(repo2, {"benchmarks/baseline.lock.json": lock}, f"chore: lock\n\n{OK_TRAILERS}")
    _git(repo2, "tag", phase_map.GATE_TAG)
    commit(repo2, {"src/manga_ar/edu/card.py": "x = 1\n"}, "feat: card\n\nPhase: 2\nStep: 2.1")
    assert fails(repo2) == []


@pytest.mark.parametrize(
    ("pyproject", "needle"),
    [
        ('[project]\ndependencies = ["spacy>=3"]\n', "spacy declared"),
        ('[project.optional-dependencies]\ndev = ["nltk"]\n', "nltk declared outside [bench]"),
        ('[project.optional-dependencies]\nedu = ["x"]\n', "extra 'edu' exists"),
        ('[project.optional-dependencies]\ngui = ["uvicorn"]\n', "uvicorn declared"),
    ],
)
def test_phase2_dependencies_fail(repo: Path, pyproject: str, needle: str) -> None:
    commit(repo, {"pyproject.toml": pyproject}, f"build: deps\n\n{OK_TRAILERS}")
    assert any(needle in m for m in fails(repo))


def test_allowed_dependencies_and_lock(repo: Path) -> None:
    ok = '[project.optional-dependencies]\nbench = ["nltk>=3.8"]\n'
    commit(repo, {"pyproject.toml": ok}, f"build: bench\n\n{OK_TRAILERS}")
    assert fails(repo) == []
    lock = '[[package]]\nname = "en-core-web-sm"\nversion = "3.8"\n'
    commit(repo, {"uv.lock": lock}, f"build: lock\n\n{OK_TRAILERS}")
    assert any("en-core-web-sm in uv.lock" in m for m in fails(repo))


def test_trailer_rules(repo: Path) -> None:
    commit(repo, {"a": "1"}, "feat: no trailers")
    commit(repo, {"b": "1"}, "feat: wrong step\n\nPhase: 1\nStep: 0.2")
    found = fails(repo)
    assert any("exactly one Phase" in m for m in found)
    assert any("does not belong to Phase 1" in m for m in found)


def test_misplaced_trailers_need_a_verified_exception(repo: Path) -> None:
    sha = commit(repo, {"a": "1"}, "feat: a\n\nPhase: 0\nStep: 0.2.3\n\nCo-Authored-By: A <a@x>")
    assert any("exactly one Phase" in m for m in fails(repo))
    entry = {"commit": sha, "subject": "feat: a", "phase": "0", "step": "0.2.3"}
    exc = repo / audit_phase_order.EXCEPTIONS
    exc.parent.mkdir(parents=True, exist_ok=True)
    exc.write_text(json.dumps({"commits": [entry]}), encoding="utf-8")
    found = audit_phase_order.audit(repo)
    assert fails(repo) == [] and any(f.warn and "acknowledged" in f.message for f in found)
    exc.write_text(json.dumps({"commits": [{**entry, "step": "0.9"}]}), encoding="utf-8")
    assert any("does not match" in m for m in fails(repo))  # negative control


def test_commit_message_and_staged_checks(repo: Path) -> None:
    assert audit_phase_order.check_message(repo, f"feat: x\n\n{OK_TRAILERS}\n") == []
    assert audit_phase_order.check_message(repo, "feat: x\n")
    bad = audit_phase_order.check_message(repo, "feat: x\n\nPhase: 2\nStep: 2.0\n")
    assert any("Phase 2 commit before tag" in f.message for f in bad)
    (repo / "data").mkdir()
    (repo / "data" / "sfx_lexicon_en_ar.tsv").write_text("bang\tبانغ\n", encoding="utf-8")
    _git(repo, "add", "data")
    assert any("data/sfx_lexicon_en_ar.tsv" in f.message
               for f in audit_phase_order.check_staged(repo))  # fmt: skip


def test_real_history_has_no_failures() -> None:
    root = Path(__file__).resolve().parents[2]
    base = audit_phase_order.base_commit(root)
    if subprocess.run(["git", "-C", str(root), "cat-file", "-e", base], check=False).returncode:
        pytest.skip("shallow clone: the baseline commit is not available")
    assert [str(f) for f in audit_phase_order.audit(root) if not f.warn] == []
