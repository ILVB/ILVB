"""Shared fixtures. Unit tests never touch the network or download models."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("MANGAAR_CACHE_DIR", ROOT / ".cache"))
os.environ.setdefault("MANGAAR_CACHE_DIR", str(CACHE))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")


@pytest.fixture(scope="session")
def cache_dir() -> Path:
    return CACHE


@pytest.fixture(scope="session")
def cjk_ready(cache_dir: Path) -> Path:
    """Skip tests that render CJK fixtures when no CJK font exists anywhere."""
    from manga_ar.errors import ModelUnavailableError
    from manga_ar.synth import find_cjk_font

    try:
        return find_cjk_font("ja", cache_dir)
    except ModelUnavailableError as exc:
        pytest.skip(f"CJK font unavailable: {exc}")


class FakeClock:
    """Deterministic clock: ``sleep`` advances time instantly and records calls."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += max(0.0, seconds)


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for key in list(os.environ):
        if key.startswith("MANGAAR"):
            monkeypatch.delenv(key, raising=False)
    yield
