"""ADR-0007 item 3: the online baseline tool refuses unreachable hosts and re-freezing."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.run_benchmark import PROFILES
from tools import freeze_online_baseline as fob


def test_refuses_without_network(tmp_path: Path) -> None:
    with pytest.raises(ConnectionError, match="unreachable"):
        fob.freeze(tmp_path / "online.json", online=False)
    assert not (tmp_path / "online.json").exists()


def test_refuses_to_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "online.json"
    target.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="frozen"):
        fob.freeze(target, online=True)


def test_profile_is_v010_google_online_only() -> None:
    cfg = PROFILES[fob.PROFILE]
    assert cfg == {"runtime.offline": False, "translate.providers": ["google"],
                   "translate.cache": True}  # fmt: skip
    assert not any(k.startswith("engine.") for k in cfg)  # valid for the v0.1.0 worktree
