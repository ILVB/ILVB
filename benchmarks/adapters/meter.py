"""Per-stage wall time and peak resident memory (sampled), for budgets (G-PERF-1)."""

from __future__ import annotations

import resource
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from benchmarks.schema import StageStats

_STATM = Path("/proc/self/statm")


def rss_mb() -> float:
    """Current resident set size in MiB (Linux /proc; else the process peak)."""
    try:
        pages = int(_STATM.read_text(encoding="ascii").split()[1])
    except OSError:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    import os

    return pages * os.sysconf("SC_PAGE_SIZE") / 2**20


@contextmanager
def measure(stats: dict[str, StageStats], stage: str, interval: float = 0.005) -> Iterator[None]:
    peak = [rss_mb()]
    stop = threading.Event()

    def sample() -> None:
        while not stop.is_set():
            peak[0] = max(peak[0], rss_mb())
            stop.wait(interval)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        stop.set()
        thread.join()
        peak[0] = max(peak[0], rss_mb())
        prev = stats.get(stage)
        seconds = elapsed + (prev.seconds if prev else 0.0)
        top = max(peak[0], prev.peak_rss_mb if prev else 0.0)
        stats[stage] = StageStats(seconds=seconds, peak_rss_mb=top)
