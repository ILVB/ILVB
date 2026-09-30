"""Heavy-weight fetcher: trust-on-first-use lock, verification and --check (no network)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "download_heavy_weights", ROOT / "scripts" / "download_heavy_weights.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(mod)
    return mod


def test_first_run_records_then_verifies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _module()
    monkeypatch.setenv("MANGAAR_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(mod, "LOCK", tmp_path / "lock.json")

    def fake(dest: Path) -> dict[str, str]:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "model.bin").write_bytes(b"weights")
        return {"repo": "x/y", "revision": "abc"}

    monkeypatch.setitem(mod.FETCHERS, "embed", fake)
    assert mod.main(["--only", "embed", "--check"]) == 1  # nothing recorded yet
    assert mod.main(["--only", "embed"]) == 0
    lock = json.loads((tmp_path / "lock.json").read_text("utf-8"))
    assert lock["embed"]["revision"] == "abc" and lock["embed"]["license"] == "Apache-2.0"
    assert set(lock["embed"]["files"]) == {"model.bin"}
    assert mod.main(["--only", "embed", "--check"]) == 0
    (tmp_path / "cache" / "models" / "heavy-embed" / "model.bin").write_bytes(b"tampered")
    assert mod.main(["--only", "embed", "--check"]) == 1  # negative control


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(SystemExit):
        _module().main(["--only", "gpt"])
