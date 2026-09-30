"""Persistent SQLite translation cache + in-run de-duplication (S5, E19)."""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import unicodedata
from pathlib import Path

from manga_ar import __version__

CACHE_VERSION = f"1:{__version__}"


def _key(provider: str, src: str, tgt: str, text: str) -> str:
    norm = " ".join(unicodedata.normalize("NFKC", text).split())
    return hashlib.sha256(f"{CACHE_VERSION}|{provider}|{src}|{tgt}|{norm}".encode()).hexdigest()


class TranslationCache:
    """Keyed by (cache version, provider, src, tgt, normalised-text hash)."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._memory: dict[str, str] = {}
        self._lock = threading.Lock()
        self._db: sqlite3.Connection | None = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(path), check_same_thread=False, timeout=10)
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS translations (key TEXT PRIMARY KEY, "
                "provider TEXT, src TEXT, tgt TEXT, source TEXT, target TEXT)"
            )
            self._db.commit()

    def get(self, provider: str, src: str, tgt: str, text: str) -> str | None:
        key = _key(provider, src, tgt, text)
        with self._lock:
            if key in self._memory:
                return self._memory[key]
            if self._db is None:
                return None
            row = self._db.execute(
                "SELECT target FROM translations WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        value = str(row[0])
        with self._lock:
            self._memory[key] = value
        return value

    def put(self, provider: str, src: str, tgt: str, text: str, target: str) -> None:
        key = _key(provider, src, tgt, text)
        with self._lock:
            self._memory[key] = target
            if self._db is not None:
                self._db.execute(
                    "INSERT OR REPLACE INTO translations VALUES (?, ?, ?, ?, ?, ?)",
                    (key, provider, src, tgt, text, target),
                )
                self._db.commit()

    def close(self) -> None:
        with self._lock:
            if self._db is not None:
                self._db.close()
                self._db = None
