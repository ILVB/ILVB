"""Cooperative cancellation (GUI cancel button, E17): checked between pages and regions."""

from __future__ import annotations

import threading

from manga_ar.errors import CancelledError


class CancelToken:
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def reset(self) -> None:
        self._event.clear()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        if self._event.is_set():
            raise CancelledError("cancelled by the user")
