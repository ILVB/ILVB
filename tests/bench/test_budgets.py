"""0.6: budgets.yaml schema, provenance and the G-PERF-1 check."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from benchmarks import budgets

ROOT = Path(__file__).resolve().parents[2]


def _perf(scale: float = 1.0, rss: float = 1000.0) -> dict[str, Any]:
    return {"stage_seconds": {s: {"p50": 0.5 * scale, "p95": 1.0 * scale, "max": 2.0}
                              for s in budgets.STAGES}, "peak_rss_mb": rss}  # fmt: skip


def test_committed_budgets_load_and_cite_the_frozen_baseline() -> None:
    b = budgets.load()
    frozen = ROOT / b.baseline["file"]
    assert hashlib.sha256(frozen.read_bytes()).hexdigest() == b.baseline["sha256"]
    for name, s in b.stages.items():
        assert s.p50 <= s.p95 and s.p50 >= s.baseline_p50, name
    assert b.memory["peak_rss_mb"].ceiling is not None


def test_violations() -> None:
    b = budgets.load()
    assert budgets.violations(_perf(), b) == []
    slow = budgets.violations(_perf(scale=100.0, rss=99999.0), b)
    assert any(v.startswith("translate p95") for v in slow) and any("peak_rss" in v for v in slow)
    partial = _perf()
    del partial["stage_seconds"]["ocr"]
    assert "ocr: not measured" in budgets.violations(partial, b)  # never a silent pass
