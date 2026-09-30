from __future__ import annotations

import io
import logging

import pytest

from manga_ar.logging_setup import configure_logging, ensure_utf8_streams, get_logger
from manga_ar.models.device import is_device_failure, resolve_device, run_with_cpu_fallback


def test_resolve_device_cpu_fallbacks() -> None:
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in {"cpu", "cuda", "mps"}
    import torch

    if not torch.cuda.is_available():
        assert resolve_device("cuda") == "cpu"


def test_cpu_fallback_on_oom() -> None:
    seen: list[str] = []

    def run(device: str) -> str:
        seen.append(device)
        if device == "cuda":
            raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
        return f"ran on {device}"

    assert run_with_cpu_fallback(run, "cuda", "lama") == ("ran on cpu", "cpu")
    assert seen == ["cuda", "cpu"]


def test_non_device_errors_propagate() -> None:
    def run(device: str) -> str:
        raise RuntimeError("shape mismatch")

    with pytest.raises(RuntimeError, match="shape"):
        run_with_cpu_fallback(run, "cuda", "x")
    assert is_device_failure(MemoryError()) and not is_device_failure(ValueError("x"))


def test_logging_arabic_cjk_on_legacy_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = io.BytesIO()
    legacy = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
    monkeypatch.setattr("sys.stderr", legacy)
    ensure_utf8_streams()
    logger = configure_logging("INFO")
    get_logger("test").info("نص عربي 日本語 한국어 ♡", extra={"ctx": {"page": "صفحة1"}})
    for h in logger.handlers:
        h.flush()
    text = raw.getvalue().decode("utf-8")
    assert "نص عربي 日本語" in text and "page=صفحة1" in text
    configure_logging("WARNING", stream=io.StringIO())
    assert logging.getLogger("manga_ar").level == logging.WARNING
