"""0.4: environment capture content and hash memoisation."""

from __future__ import annotations

from pathlib import Path

from benchmarks import env


def test_capture_keys_and_memo(tmp_path: Path) -> None:
    data = env.capture()
    assert {"os", "cpu", "ram_gb", "gpu", "python", "packages", "git", "models"} <= set(data)
    assert data["packages"]["numpy"] and len(data["git"]["commit"]) == 40
    f = tmp_path / "w.bin"
    f.write_bytes(b"abc")
    memo = tmp_path / "memo.json"
    first = env._hash_files([f], memo)
    assert first[str(f)] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert env._hash_files([f], memo) == first and memo.is_file()
