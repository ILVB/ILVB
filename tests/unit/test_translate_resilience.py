"""Resilience primitives with a fake clock (no real sleeping except tiny deadline waits)."""

from __future__ import annotations

import random
import threading
import time

import pytest

from manga_ar.errors import CircuitOpenError, DeadlineExceededError, ProviderError, RateLimitError
from manga_ar.translate.resilience import (
    Backoff,
    BreakerState,
    CircuitBreaker,
    ResilientCaller,
    TokenBucket,
    call_with_deadline,
)
from tests.conftest import FakeClock


def test_token_bucket_rate_and_burst(fake_clock: FakeClock) -> None:
    bucket = TokenBucket(rate=1.0, burst=2, clock=fake_clock)
    assert bucket.acquire() == 0.0 and bucket.acquire() == 0.0  # burst
    waited = bucket.acquire()
    assert waited == pytest.approx(1.0)  # third token needs one refill period
    fake_clock.sleep(10)
    assert bucket.acquire() == 0.0 and bucket.acquire() == 0.0
    with pytest.raises(ValueError):
        TokenBucket(0, 1, fake_clock)


def test_backoff_full_jitter_and_retry_after() -> None:
    b = Backoff(base=1.0, factor=2.0, cap=60.0, max_attempts=6)
    rng = random.Random(1)
    for attempt in range(1, 10):
        d = b.delay(attempt, rng)
        assert 0.0 <= d <= min(60.0, 2 ** (attempt - 1))
    assert b.delay(1, rng, retry_after=7.0) >= 7.0
    assert b.delay(1, rng, retry_after=500.0) == 60.0  # capped


def test_breaker_open_half_open_close(fake_clock: FakeClock) -> None:
    br = CircuitBreaker(threshold=5, cooldown=120.0, clock=fake_clock)
    for _ in range(4):
        br.record_failure()
    assert br.allow() and br.state == BreakerState.CLOSED
    br.record_failure()
    assert br.state == BreakerState.OPEN and not br.allow()
    fake_clock.sleep(119)
    assert not br.allow()
    fake_clock.sleep(1)
    assert br.allow() and br.state == BreakerState.HALF_OPEN
    assert not br.allow()  # only one probe in flight
    br.record_failure()
    assert br.state == BreakerState.OPEN
    fake_clock.sleep(120)
    assert br.allow()
    br.record_success()
    assert br.state == BreakerState.CLOSED and br.failures == 0


def test_deadline_wrapper_bounds_hung_calls() -> None:
    release = threading.Event()
    start = time.perf_counter()
    with pytest.raises(DeadlineExceededError):
        call_with_deadline(lambda: release.wait(30), timeout=0.05)
    assert time.perf_counter() - start < 1.0
    release.set()
    assert call_with_deadline(lambda: "ok", 1.0) == "ok"
    with pytest.raises(KeyError):
        call_with_deadline(lambda: {}["x"], 1.0)


def _caller(clock: FakeClock, attempts: int = 6, deadline: float = 1.0) -> ResilientCaller:
    return ResilientCaller(
        "fake",
        limiter=TokenBucket(100.0, 100, clock),
        breaker=CircuitBreaker(5, 120.0, clock),
        backoff=Backoff(1.0, 2.0, 60.0, attempts),
        deadline=deadline,
        clock=clock,
        rng=random.Random(0),
    )


def test_retries_then_succeeds(fake_clock: FakeClock) -> None:
    outcomes = [ProviderError("reset"), ProviderError("reset"), "مرحبا"]

    def fn() -> str:
        out = outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out

    caller = _caller(fake_clock)
    assert caller.call(fn) == "مرحبا" and caller.calls == 3
    assert len(fake_clock.sleeps) == 2 and fake_clock.sleeps[1] <= 2.0


def test_429_honours_retry_after(fake_clock: FakeClock) -> None:
    outcomes: list[object] = [RateLimitError("429", retry_after=7.0), "ok"]

    def fn() -> str:
        out = outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return str(out)

    assert _caller(fake_clock).call(fn) == "ok"
    assert fake_clock.sleeps and fake_clock.sleeps[0] >= 7.0


def test_non_retryable_stops_immediately(fake_clock: FakeClock) -> None:
    calls = []

    def fn() -> str:
        calls.append(1)
        raise ProviderError("bad language", retryable=False)

    with pytest.raises(ProviderError):
        _caller(fake_clock).call(fn)
    assert len(calls) == 1 and fake_clock.sleeps == []


def test_breaker_short_circuits_after_failures(fake_clock: FakeClock) -> None:
    caller = _caller(fake_clock, attempts=6)
    calls = []

    def fn() -> str:
        calls.append(1)
        raise ProviderError("down")

    with pytest.raises((ProviderError, CircuitOpenError)):
        caller.call(fn)
    assert len(calls) == 5  # breaker opened after 5 consecutive failures
    with pytest.raises(CircuitOpenError):
        caller.call(fn)
    assert len(calls) == 5  # not attempted while open


def test_deadline_inside_caller_is_retried(fake_clock: FakeClock) -> None:
    release = threading.Event()
    state = {"n": 0}

    def fn() -> str:
        state["n"] += 1
        if state["n"] == 1:
            release.wait(30)  # hang: must be abandoned by the deadline
        return "done"

    caller = _caller(fake_clock, deadline=0.05)
    try:
        assert caller.call(fn) == "done" and state["n"] == 2
    finally:
        release.set()
