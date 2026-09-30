"""Seeded bootstrap confidence intervals and latency percentiles."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

T = float


def bootstrap_ci(
    items: Sequence[object], statistic: Callable[[Sequence[object]], float],
    n: int = 1000, seed: int = 20260930, alpha: float = 0.05,
) -> tuple[float, float]:  # fmt: skip
    """Percentile bootstrap CI of ``statistic`` over resampled ``items`` (fixed seed)."""
    if not items:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    k = len(items)
    values = [statistic([items[i] for i in rng.integers(0, k, k)]) for _ in range(n)]
    return (float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2)))


def percentiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"p50": float("nan"), "p95": float("nan"), "max": float("nan")}
    arr = np.asarray(values, dtype=np.float64)
    return {"p50": float(np.percentile(arr, 50)), "p95": float(np.percentile(arr, 95)),
            "max": float(arr.max())}  # fmt: skip
