"""UTF-8-safe, structured logging.

Arabic and CJK text in log records must never raise ``UnicodeEncodeError`` on legacy
Windows consoles, so stdout/stderr are reconfigured to UTF-8 with ``errors="replace"``.
Context passed via ``extra={"ctx": {...}}`` is rendered as ``key=value`` pairs.
"""

from __future__ import annotations

import io
import logging
import sys
from typing import Any, TextIO

_CONFIGURED_HANDLER: logging.Handler | None = None


def ensure_utf8_streams() -> None:
    """Reconfigure stdout/stderr to UTF-8 with replacement for unencodable characters."""
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, io.UnsupportedOperation):
                # Detached or non-reconfigurable stream (e.g. pytest capture): leave it.
                continue


class ContextFormatter(logging.Formatter):
    """``LEVEL logger: message key=value ...`` with optional ``ctx`` mapping."""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        ctx: Any = getattr(record, "ctx", None)
        if isinstance(ctx, dict) and ctx:
            pairs = " ".join(f"{k}={v}" for k, v in ctx.items())
            return f"{base} | {pairs}"
        return base


def configure_logging(level: str = "INFO", stream: TextIO | None = None) -> logging.Logger:
    """Configure the ``manga_ar`` logger once (idempotent) and return it."""
    global _CONFIGURED_HANDLER
    ensure_utf8_streams()
    logger = logging.getLogger("manga_ar")
    logger.setLevel(level.upper())
    if _CONFIGURED_HANDLER is not None:
        logger.removeHandler(_CONFIGURED_HANDLER)
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(ContextFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    _CONFIGURED_HANDLER = handler
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger of ``manga_ar``."""
    return logging.getLogger(name if name.startswith("manga_ar") else f"manga_ar.{name}")
