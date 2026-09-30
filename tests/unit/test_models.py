"""Model manager with a fake HTTP session (no network)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from manga_ar.errors import ModelUnavailableError
from manga_ar.models import manager as manager_mod
from manga_ar.models.manager import ModelManager
from manga_ar.models.registry import REGISTRY, ModelSpec

PAYLOAD = b"model-bytes-" * 1000


class FakeResponse:
    def __init__(self, status: int, body: bytes) -> None:
        self.status_code = status
        self.headers: dict[str, str] = {}
        self._body = body

    def iter_content(self, chunk_size: int) -> Any:
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i : i + chunk_size]

    def close(self) -> None:
        pass


class FakeSession:
    def __init__(self, script: list[Any]) -> None:
        self.script = script
        self.calls: list[dict[str, str]] = []

    def get(self, url: str, *, headers: dict[str, str], stream: bool, timeout: Any) -> Any:
        self.calls.append(dict(headers))
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


@pytest.fixture
def fake_spec(monkeypatch: pytest.MonkeyPatch) -> ModelSpec:
    spec = ModelSpec(
        name="fake",
        kind="file",
        description="t",
        license="MIT",
        url="https://example.invalid/m.bin",
        sha256=hashlib.sha256(PAYLOAD).hexdigest(),
        size=len(PAYLOAD),
        filename="fake/m.bin",
    )
    monkeypatch.setitem(REGISTRY, "fake", spec)
    return spec


def _mgr(tmp_path: Path, script: list[Any], **kw: Any) -> tuple[ModelManager, FakeSession]:
    session = FakeSession(script)
    sleeps: list[float] = []
    mgr = ModelManager(tmp_path, session=session, sleep=sleeps.append, **kw)
    return mgr, session


def test_download_verify_and_cache(tmp_path: Path, fake_spec: ModelSpec) -> None:
    mgr, session = _mgr(tmp_path, [FakeResponse(200, PAYLOAD)])
    path = mgr.ensure("fake")
    assert path.read_bytes() == PAYLOAD and mgr.is_present("fake")
    assert mgr.ensure("fake") == path and len(session.calls) == 1  # cached, no 2nd call
    assert mgr.verify("fake")[0]


def test_resume_partial_download(tmp_path: Path, fake_spec: ModelSpec) -> None:
    part = tmp_path / "models" / "fake" / "m.bin.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(PAYLOAD[:5000])
    mgr, session = _mgr(tmp_path, [FakeResponse(206, PAYLOAD[5000:])])
    assert mgr.ensure("fake").read_bytes() == PAYLOAD
    assert session.calls[0] == {"Range": "bytes=5000-"}


def test_checksum_mismatch_then_retry_success(tmp_path: Path, fake_spec: ModelSpec) -> None:
    mgr, session = _mgr(
        tmp_path,
        [FakeResponse(200, b"corrupt" * 10), ConnectionError("reset"), FakeResponse(200, PAYLOAD)],
    )
    assert mgr.ensure("fake").read_bytes() == PAYLOAD
    assert len(session.calls) == 3


def test_retries_exhausted_gives_instructions(tmp_path: Path, fake_spec: ModelSpec) -> None:
    mgr, _ = _mgr(tmp_path, [FakeResponse(503, b"")] * 4, max_attempts=4)
    with pytest.raises(ModelUnavailableError, match="install it manually"):
        mgr.ensure("fake")


def test_offline_never_calls_network(tmp_path: Path, fake_spec: ModelSpec) -> None:
    mgr, session = _mgr(tmp_path, [], offline=True)
    with pytest.raises(ModelUnavailableError, match="offline"):
        mgr.ensure("fake")
    assert session.calls == []
    with pytest.raises(ModelUnavailableError, match="offline"):
        mgr.ensure("mt-ja-en")


def test_low_disk(tmp_path: Path, fake_spec: ModelSpec, monkeypatch: pytest.MonkeyPatch) -> None:
    class Usage:
        free = 10

    monkeypatch.setattr(manager_mod.shutil, "disk_usage", lambda p: Usage())
    mgr, session = _mgr(tmp_path, [])
    with pytest.raises(ModelUnavailableError, match="disk space"):
        mgr.ensure("fake")
    assert session.calls == []


def test_corrupted_cache_redownloads(tmp_path: Path, fake_spec: ModelSpec) -> None:
    dest = tmp_path / "models" / "fake" / "m.bin"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"x" * len(PAYLOAD))  # right size, wrong content
    mgr, session = _mgr(tmp_path, [FakeResponse(200, PAYLOAD)])
    assert mgr.ensure("fake").read_bytes() == PAYLOAD and len(session.calls) == 1


def test_lazy_load_once(tmp_path: Path, fake_spec: ModelSpec) -> None:
    mgr, _ = _mgr(tmp_path, [FakeResponse(200, PAYLOAD)])
    calls: list[Path] = []

    def loader(p: Path) -> str:
        calls.append(p)
        return "model"

    assert mgr.load("fake", loader) == "model" and mgr.load("fake", loader) == "model"
    assert len(calls) == 1
    mgr.unload("fake")
    mgr.load("fake", loader)
    assert len(calls) == 2


def test_unknown_and_status(tmp_path: Path) -> None:
    mgr = ModelManager(tmp_path, offline=True)
    with pytest.raises(ModelUnavailableError, match="unknown model"):
        mgr.ensure("nope")
    names = {row["name"] for row in mgr.status()}
    assert {"lama", "ctd", "easyocr", "manga-ocr", "mt-en-ar"} <= names
    ctd = next(r for r in mgr.status() if r["name"] == "ctd")
    assert ctd["copyleft_or_nc"] is True
