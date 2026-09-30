"""Adapter plumbing that needs no models: stage meter and worker environment."""

from __future__ import annotations

import os

import pytest

from benchmarks.adapters import client
from benchmarks.adapters.meter import measure, rss_mb
from benchmarks.schema import StageStats


def test_meter_accumulates_time_and_peak() -> None:
    stats: dict[str, StageStats] = {}
    with measure(stats, "stage"):
        block = bytearray(20 * 2**20)  # 20 MiB touched inside the stage
        block[::4096] = b"x" * len(block[::4096])
    first = stats["stage"]
    with measure(stats, "stage"):
        pass
    assert stats["stage"].seconds >= first.seconds and first.peak_rss_mb >= rss_mb() - 64
    del block


def test_worker_env_puts_candidate_code_first() -> None:
    env = client.worker_env("candidate")
    first = env["PYTHONPATH"].split(os.pathsep)[0]
    assert first.endswith("src") and ".baseline" not in first
    assert env["HF_HUB_OFFLINE"] == "1" and env["PYTHONHASHSEED"] == "0"
    with pytest.raises(ValueError, match="unknown"):
        client.code_dir("nightly")
