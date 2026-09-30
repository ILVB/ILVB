"""0.2.4: the frozen baseline is immutable and hash-verified (no models needed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools import freeze_baseline


def test_refuses_to_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "baseline.json"
    target.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="frozen"):
        freeze_baseline.freeze(target)


def test_verify_detects_tampering(tmp_path: Path) -> None:
    import hashlib

    target = tmp_path / "baseline.json"
    target.write_bytes(b'{"a": 1}\n')
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    (tmp_path / "baseline.json.sha256").write_text(f"{digest}  baseline.json\n", "utf-8")
    assert freeze_baseline.verify(target)
    target.write_bytes(b'{"a": 2}\n')  # negative control: an edited number
    assert not freeze_baseline.verify(target)
