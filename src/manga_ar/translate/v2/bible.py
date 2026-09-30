"""Series Bible store (v2, step 1.1.6 core; E-17).

One SQLite file per series under ``<root>/<slug>.sqlite3``, so data cannot leak between
series. Writes happen inside ``writer()``, which holds an exclusive file lock (a second
writer, in any process, gets SeriesBusyError); readers are never blocked (WAL).
``snapshot()`` versions the store and ``rollback()`` restores a version. ``reset()`` keeps
a snapshot first, so it can be undone. On open, a store that fails SQLite's integrity check
is quarantined and restored from its latest snapshot, or recreated empty.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from filelock import FileLock, Timeout

from manga_ar.errors import TranslationError
from manga_ar.logging_setup import get_logger

log = get_logger(__name__)
SCHEMA = """
CREATE TABLE IF NOT EXISTS glossary (term TEXT PRIMARY KEY, arabic TEXT NOT NULL,
                                     locked INTEGER NOT NULL DEFAULT 1);
CREATE TABLE IF NOT EXISTS characters (canonical TEXT PRIMARY KEY, arabic TEXT NOT NULL,
                                       gender TEXT, register TEXT, aliases TEXT DEFAULT '');
CREATE TABLE IF NOT EXISTS memory (seq INTEGER PRIMARY KEY AUTOINCREMENT, page TEXT,
                                   region TEXT, source TEXT, target TEXT, speaker TEXT);
"""


class SeriesBusyError(TranslationError):
    """Another writer holds this series' store."""


def slug(series_id: str) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", series_id).strip("._")[:40] or "series"
    return f"{base}-{hashlib.sha256(series_id.encode('utf-8')).hexdigest()[:10]}"


class SeriesBible:
    def __init__(self, root: Path, series_id: str, lock_timeout: float = 5.0) -> None:
        self.series_id = series_id
        self.root = root
        self.path = root / f"{slug(series_id)}.sqlite3"
        self.lock = FileLock(str(self.path) + ".lock", timeout=lock_timeout)
        self.recovered = False
        root.mkdir(parents=True, exist_ok=True)
        self._open()

    # ---------------------------------------------------------------- storage
    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path)
        con.execute("PRAGMA journal_mode=WAL")
        con.executescript(SCHEMA)
        return con

    def _healthy(self) -> bool:
        if not self.path.exists():
            return True
        try:
            with sqlite3.connect(self.path) as con:
                return bool(con.execute("PRAGMA integrity_check").fetchone()[0] == "ok")
        except sqlite3.DatabaseError:
            return False

    def _open(self) -> None:
        if not self._healthy():
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
            quarantine = self.path.with_name(f"{self.path.name}.corrupt-{stamp}")
            self.path.replace(quarantine)
            for sidecar in (self.path.with_name(self.path.name + s) for s in ("-wal", "-shm")):
                sidecar.unlink(missing_ok=True)
            versions = self.versions()
            if versions:
                self._restore(versions[-1])
            outcome = f"restored v{versions[-1]}" if versions else "reset"
            log.warning("series store %s was corrupt; quarantined as %s, %s",
                        self.path.name, quarantine.name, outcome)  # fmt: skip
            self.recovered = True
        self._connect().close()

    @contextmanager
    def writer(self) -> Iterator[sqlite3.Connection]:
        try:
            self.lock.acquire()
        except Timeout as exc:
            raise SeriesBusyError(f"series {self.series_id!r} is being written elsewhere") from exc
        try:
            con = self._connect()
            try:
                with con:
                    yield con
            finally:
                con.close()
        finally:
            self.lock.release()

    def _read(self, sql: str, args: tuple[object, ...] = ()) -> list[tuple[object, ...]]:
        con = self._connect()
        try:
            return list(con.execute(sql, args))
        finally:
            con.close()

    # ---------------------------------------------------------------- content
    def lock_term(self, term: str, arabic: str) -> None:
        with self.writer() as con:
            con.execute("INSERT OR REPLACE INTO glossary VALUES (?, ?, 1)", (term, arabic))

    def glossary(self) -> dict[str, str]:
        return {str(t): str(a) for t, a in self._read("SELECT term, arabic FROM glossary")}

    def remember(self, page: str, region: str, source: str, target: str,
                 speaker: str | None = None) -> None:  # fmt: skip
        with self.writer() as con:
            con.execute(
                "INSERT INTO memory (page, region, source, target, speaker) VALUES (?, ?, ?, ?, ?)",
                (page, region, source, target, speaker),
            )

    def recent(self, n: int) -> list[tuple[str, str]]:
        rows = self._read("SELECT source, target FROM memory ORDER BY seq DESC LIMIT ?", (n,))
        return [(str(s), str(t)) for s, t in reversed(rows)]

    # --------------------------------------------------------------- versions
    def _version_path(self, version: int) -> Path:
        return self.path.with_name(f"{self.path.stem}.v{version}.sqlite3")

    def versions(self) -> list[int]:
        pattern = re.compile(rf"^{re.escape(self.path.stem)}\.v(\d+)\.sqlite3$")
        found = (pattern.match(p.name) for p in self.root.iterdir())
        return sorted(int(m.group(1)) for m in found if m)

    def snapshot(self) -> int:
        version = (self.versions() or [0])[-1] + 1
        with self.writer() as con, sqlite3.connect(self._version_path(version)) as dst:
            con.backup(dst)
        return version

    def _restore(self, version: int) -> None:
        with sqlite3.connect(self._version_path(version)) as src:
            dst = sqlite3.connect(self.path)
            try:
                src.backup(dst)
            finally:
                dst.close()

    def rollback(self, version: int) -> None:
        if version not in self.versions():
            raise TranslationError(f"series {self.series_id!r} has no version {version}")
        with self.writer():
            self._restore(version)

    def reset(self) -> int:
        """Empty the store after taking a snapshot; returns that snapshot's version."""
        version = self.snapshot()
        with self.writer() as con:
            con.executescript("DELETE FROM glossary; DELETE FROM characters; DELETE FROM memory;")
        return version
