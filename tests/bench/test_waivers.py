"""Waivers: scope, lifting and the rule that a waiver never produces PASS."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks import waivers


def test_committed_waiver_scope() -> None:
    (w,) = waivers.load()
    assert w.id == "W-HEAVY-1" and w.status == "PENDING-HOST-DOWNLOAD"
    assert set(w.checks) == {"G-TR-1", "G-TR-2", "G-TR-3", "G-TR-4", "G-INP-2/LPIPS"}


def test_decide(tmp_path: Path) -> None:
    ws = waivers.load()
    lock = tmp_path / "lock.json"
    assert waivers.decide("G-TR-1", None, ws, lock) == (
        "WAIVED", "PENDING-HOST-DOWNLOAD (waived: W-HEAVY-1, 2026-09-30)")  # fmt: skip
    assert waivers.decide("G-TYPE-1", None, ws, lock) == ("FAIL", "NOT-MEASURABLE")
    assert waivers.decide("G-TR-1", False, ws, lock) == ("FAIL", "")  # measured wins
    assert waivers.decide("G-TR-1", True, ws, lock) == ("PASS", "")
    lock.write_text(json.dumps({m: {} for m in ws[0].models}), encoding="utf-8")
    assert waivers.decide("G-TR-1", None, ws, lock) == ("FAIL", "NOT-MEASURABLE")  # lifted
