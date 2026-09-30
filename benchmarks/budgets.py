"""Performance budgets (benchmarks/budgets.yaml) and the G-PERF-1 check."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

BUDGETS = Path(__file__).resolve().parent / "budgets.yaml"
STAGES = ("detect", "ocr", "inpaint", "typeset", "translate")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StageBudget(_Strict):
    baseline_p50: float
    baseline_p95: float
    p50: float
    p95: float


class MemoryBudget(_Strict):
    baseline: float | None
    ceiling: float | None


class Budgets(_Strict):
    host: dict[str, Any]
    baseline: dict[str, Any]
    stages: dict[str, StageBudget]
    memory: dict[str, MemoryBudget]


def load(path: Path = BUDGETS) -> Budgets:
    data = yaml.safe_load(path.read_text("utf-8"))
    if data.pop("schema", None) != 1:
        raise ValueError(f"{path}: unsupported schema (expected 1)")
    budgets = Budgets.model_validate(data)
    missing = set(STAGES) - set(budgets.stages)
    if missing:
        raise ValueError(f"budgets.yaml lacks stages {sorted(missing)}")
    return budgets


def violations(perf: dict[str, Any], budgets: Budgets) -> list[str]:
    """G-PERF-1: each measured stage p50/p95 and the peak RSS within the ceilings.

    ``perf`` is a result summary's ``perf`` block (benchmarks.summary.perf_summary).
    A stage that was not measured is a violation, never a silent pass.
    """
    out = []
    for stage, budget in budgets.stages.items():
        measured = perf["stage_seconds"].get(stage)
        if measured is None:
            out.append(f"{stage}: not measured")
            continue
        for q in ("p50", "p95"):
            if measured[q] > getattr(budget, q):
                out.append(f"{stage} {q} {measured[q]:.2f}s > {getattr(budget, q):.2f}s")
    rss, ceiling = perf.get("peak_rss_mb"), budgets.memory["peak_rss_mb"].ceiling
    if rss is None:
        out.append("peak_rss_mb: not measured")
    elif ceiling is not None and rss > ceiling:
        out.append(f"peak_rss_mb {rss:.0f} > {ceiling:.0f}")
    return out
