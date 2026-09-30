"""Translator interface. A provider performs ONE request (``translate``); batching,
failover, caching and validation live in :mod:`manga_ar.translate.service`."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Translator(Protocol):
    name: str
    network: bool  # needs the internet (skipped in offline mode)

    def available(self) -> bool:
        """Cheap readiness check (installed, configured, model present or obtainable)."""
        ...

    def supports(self, src: str, tgt: str) -> bool: ...

    def translate(self, text: str, src: str, tgt: str) -> str:
        """Translate one text. Raises :class:`ProviderError` (retryable or not)."""
        ...
