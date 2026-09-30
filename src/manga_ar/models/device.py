"""Device selection with automatic CPU fallback (E12)."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from typing import TypeVar

from manga_ar.logging_setup import get_logger

log = get_logger(__name__)
T = TypeVar("T")
_WARNED: set[str] = set()


def torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def resolve_device(requested: str) -> str:
    """Map ``auto|cpu|cuda|mps`` to an available device, falling back to ``cpu``."""
    if requested == "cpu" or not torch_available():
        if requested not in {"cpu", "auto"}:
            _warn_once(f"{requested}-no-torch", f"device {requested!r} needs torch; using cpu")
        return "cpu"
    import torch

    cuda = bool(torch.cuda.is_available())
    mps_backend = getattr(torch.backends, "mps", None)
    mps = bool(mps_backend is not None and mps_backend.is_available())
    if requested == "auto":
        return "cuda" if cuda else ("mps" if mps else "cpu")
    if requested == "cuda" and cuda:
        return "cuda"
    if requested == "mps" and mps:
        return "mps"
    _warn_once(f"{requested}-missing", f"device {requested!r} unavailable; using cpu")
    return "cpu"


def is_device_failure(exc: BaseException) -> bool:
    """Out-of-memory or accelerator runtime failures that justify a CPU retry."""
    if isinstance(exc, MemoryError):
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        key in text
        for key in ("out of memory", "cuda", "cudnn", "mps", "hip", "device-side", "cublas")
    )


def free_accelerator_memory() -> None:
    if not torch_available():
        return
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mps_backend = getattr(torch, "mps", None)
    empty = getattr(mps_backend, "empty_cache", None)
    if callable(empty):
        try:
            empty()
        except RuntimeError as exc:  # MPS not initialised on this machine
            log.debug("mps.empty_cache failed: %s", exc)


def run_with_cpu_fallback(run: Callable[[str], T], device: str, label: str) -> tuple[T, str]:
    """Run ``run(device)``; on OOM/accelerator failure free caches and retry on CPU."""
    try:
        return run(device), device
    except (RuntimeError, MemoryError, OSError) as exc:
        if device == "cpu" or not is_device_failure(exc):
            raise
        free_accelerator_memory()
        _warn_once(f"{label}-fallback", f"{label}: {device} failed ({exc}); retrying on cpu")
        return run("cpu"), "cpu"


def _warn_once(key: str, message: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        log.warning(message)
