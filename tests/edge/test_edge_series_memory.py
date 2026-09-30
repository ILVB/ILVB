"""E-17: Series Bible store isolation, reset/rollback, corruption recovery, single writer."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

from manga_ar.translate.v2.bible import SeriesBible, SeriesBusyError

ROOT = Path(__file__).resolve().parents[2]


def test_e17_series_are_isolated(tmp_path: Path) -> None:
    a, b = SeriesBible(tmp_path, "One Piece"), SeriesBible(tmp_path, "Naruto")
    a.lock_term("ルフィ", "لوفي")
    a.remember("p1", "r1", "行くぞ", "هيا بنا", "ルフィ")
    assert a.glossary() == {"ルフィ": "لوفي"} and a.recent(5) == [("行くぞ", "هيا بنا")]
    assert b.glossary() == {} and b.recent(5) == []  # zero cross-series leakage
    assert a.path != b.path and SeriesBible(tmp_path, "One Piece").glossary() == a.glossary()


def test_e17_reset_and_rollback(tmp_path: Path) -> None:
    bible = SeriesBible(tmp_path, "s")
    bible.lock_term("先輩", "سينباي")
    v1 = bible.snapshot()
    bible.lock_term("先生", "المعلم")
    bible.rollback(v1)
    assert bible.glossary() == {"先輩": "سينباي"}
    undo = bible.reset()
    assert bible.glossary() == {}
    bible.rollback(undo)  # a reset is itself reversible
    assert bible.glossary() == {"先輩": "سينباي"}


def test_e17_corrupted_store_is_quarantined_and_restored(tmp_path: Path) -> None:
    bible = SeriesBible(tmp_path, "s")
    bible.lock_term("先輩", "سينباي")
    bible.snapshot()
    bible.lock_term("lost", "مفقود")
    bible.path.write_bytes(b"this is not a sqlite database" * 100)
    for side in ("-wal", "-shm"):
        bible.path.with_name(bible.path.name + side).unlink(missing_ok=True)
    reopened = SeriesBible(tmp_path, "s")
    assert reopened.recovered and reopened.glossary() == {"先輩": "سينباي"}
    assert len(list(tmp_path.glob("*.corrupt-*"))) == 1  # evidence kept, not deleted
    fresh_root = tmp_path / "fresh"
    first = SeriesBible(fresh_root, "t")
    first.path.write_bytes(b"\x00" * 4096)
    empty = SeriesBible(fresh_root, "t")  # no snapshot to restore: recreated empty
    assert empty.recovered and empty.glossary() == {}


def test_e17_single_writer_across_processes(tmp_path: Path) -> None:
    holder = subprocess.Popen(
        [sys.executable, "-c",
         "import sys, time; from pathlib import Path; "
         "from manga_ar.translate.v2.bible import SeriesBible; "
         "b = SeriesBible(Path(sys.argv[1]), 's'); w = b.writer(); w.__enter__(); "
         "print('held', flush=True); time.sleep(30)", str(tmp_path)],
        stdout=subprocess.PIPE, text=True, cwd=ROOT,
    )  # fmt: skip
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == "held"
        contender = SeriesBible(tmp_path, "s", lock_timeout=0.2)
        with pytest.raises(SeriesBusyError):
            contender.lock_term("x", "س")
        assert contender.glossary() == {}  # readers are not blocked
    finally:
        holder.kill()
        holder.wait()
    time.sleep(0.1)
    SeriesBible(tmp_path, "s", lock_timeout=2.0).lock_term("x", "س")  # free again
