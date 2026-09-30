"""0.2.1: the baseline worktree is pinned to the frozen v0.1.0 commit."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools import baseline_worktree as bw


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    (root / "a.txt").write_text("v1", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "one")
    _git(root, "tag", "v0.1.0")
    sha = _git(root, "rev-parse", "HEAD")
    (root / "a.txt").write_text("v2", encoding="utf-8")
    _git(root, "commit", "-qam", "two")
    return root, sha


def test_create_and_verify(repo: tuple[Path, str]) -> None:
    root, sha = repo
    lock = bw.BaselineLock(tag="v0.1.0", commit=sha, worktree=".baseline")
    bw.ensure(root, lock)
    assert (root / ".baseline" / "a.txt").read_text(encoding="utf-8") == "v1"
    bw.verify(root, lock)  # idempotent
    bw.ensure(root, lock)


def test_moved_tag_is_refused(repo: tuple[Path, str]) -> None:
    root, sha = repo
    _git(root, "tag", "-f", "v0.1.0", "HEAD")  # someone moved the tag
    with pytest.raises(bw.BaselineError, match="tag"):
        bw.ensure(root, bw.BaselineLock(tag="v0.1.0", commit=sha, worktree=".baseline"))


def test_dirty_or_moved_worktree_is_refused(repo: tuple[Path, str]) -> None:
    root, sha = repo
    lock = bw.BaselineLock(tag="v0.1.0", commit=sha, worktree=".baseline")
    bw.ensure(root, lock)
    (root / ".baseline" / "a.txt").write_text("tampered", encoding="utf-8")
    with pytest.raises(bw.BaselineError, match="dirty"):
        bw.verify(root, lock)
    _git(root / ".baseline", "checkout", "-q", "-f", "a.txt")
    _git(root / ".baseline", "checkout", "-q", "--detach", "main")
    with pytest.raises(bw.BaselineError, match="HEAD"):
        bw.verify(root, lock)


def test_lock_file_parses() -> None:
    lock = bw.read_lock(bw.ROOT / "benchmarks" / "baseline.lock.json")
    assert lock.tag == "v0.1.0" and len(lock.commit) == 40
