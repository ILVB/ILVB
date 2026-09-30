"""0.5.5: G-LIC-1 register check on a throwaway repo, with negative controls."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tools import check_licenses


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          check=True).stdout.strip()  # fmt: skip


def _lock(*names: str) -> str:
    return "".join(f'[[package]]\nname = "{n}"\nversion = "1.0"\n\n' for n in names)


def test_new_dependencies_must_be_registered(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "T")
    (tmp_path / "uv.lock").write_text(_lock("numpy"), "utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")
    base = _git(tmp_path, "rev-parse", "HEAD")
    (tmp_path / "benchmarks").mkdir()
    lock = {"tag": "v0.1.0", "commit": base, "worktree": ".baseline"}
    (tmp_path / "benchmarks" / "baseline.lock.json").write_text(json.dumps(lock), "utf-8")
    (tmp_path / "uv.lock").write_text(_lock("numpy", "sacrebleu", "Jiwer"), "utf-8")
    (tmp_path / "docs").mkdir()
    register = tmp_path / "docs" / "LICENSES.md"
    table = "| Package | Version | Licence |\n|---|---|---|\n| sacrebleu | 2.6 | Apache-2.0 |\n"
    register.write_text(table, "utf-8")
    assert check_licenses.problems(tmp_path) == [
        "jiwer: introduced since v0.1.0 but not in docs/LICENSES.md"
    ]
    register.write_text(table + "| jiwer | 4.0 | Apache-2.0 |\n", "utf-8")
    assert check_licenses.problems(tmp_path) == []
    nc = "| model-x | CC-BY-NC-4.0 | bundled | **non-commercial** | default |\n"
    register.write_text(table + "| jiwer | 4.0 | Apache-2.0 |\n" + nc, "utf-8")
    assert any("non-commercial" in p for p in check_licenses.problems(tmp_path))
