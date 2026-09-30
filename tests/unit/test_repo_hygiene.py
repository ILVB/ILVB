"""Repository hygiene: every source file is committable, no weights or big binaries.

Regression: an unanchored ``models/`` ignore pattern (meant for weight folders) silently
excluded the ``manga_ar.models`` package; only the clean-install test noticed.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or not (ROOT / ".git").exists(), reason="needs a git checkout"
)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout


def test_no_source_file_is_ignored() -> None:
    sources = [
        p.relative_to(ROOT).as_posix()
        for top in ("src", "tests", "scripts", "configs", "packaging", "docs")
        for p in (ROOT / top).rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and ".egg-info" not in str(p)
        and not str(p.relative_to(ROOT)).startswith("tests/fixtures/fonts")
    ]
    ignored = _git("check-ignore", *sources).split()
    assert ignored == [], ignored


def test_no_large_or_weight_files_tracked() -> None:
    tracked = _git("ls-files", "-z").split("\0")
    weights = {".pt", ".pth", ".onnx", ".ckpt", ".safetensors", ".bin"}
    big = []
    for name in filter(None, tracked):
        path = ROOT / name
        too_big = path.is_file() and path.stat().st_size > 2_000_000 and path.suffix != ".ttf"
        if path.suffix.lower() in weights or too_big:
            big.append(name)
    assert big == [], big
