"""Resilience primitives for network translation (S5, E1/E2).

- :class:`Clock` abstraction (real or fake) so tests never sleep.
- :class:`TokenBucket` client-side rate limiter.
- :class:`Backoff` exponential backoff with full jitter, honouring ``Retry-After``.
- :class:`CircuitBreaker` per provider (closed → open → half-open → closed).
- :func:`call_with_deadline` hard deadline around calls that expose no timeout.
- :class:`ResilientCaller` combining all of the above.
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, TypeVar

from manga_ar.errors import CircuitOpenError, DeadlineExceededError, ProviderError
from manga_ar.logging_setup import get_logger

log = get_logger(__name__)
T = TypeVar("T")


class Clock(Protocol):
    def time(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


class RealClock:
    def time(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


class TokenBucket:
    """``rate`` tokens per second, up to ``burst``; ``acquire`` blocks via the clock."""

    def __init__(self, rate: float, burst: int, clock: Clock) -> None:
        if rate <= 0 or burst < 1:
            raise ValueError("rate must be > 0 and burst >= 1")
        self.rate = rate
        self.capacity = float(burst)
        self.clock = clock
        self.tokens = float(burst)
        self.stamp = clock.time()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = self.clock.time()
        self.tokens = min(self.capacity, self.tokens + (now - self.stamp) * self.rate)
        self.stamp = now

    def acquire(self) -> float:
        """Take one token, sleeping as needed. Returns the time waited."""
        waited = 0.0
        while True:
            with self._lock:
                self._refill()
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return waited
                wait = (1.0 - self.tokens) / self.rate
            self.clock.sleep(wait)
            waited += wait


@dataclass(frozen=True)
class Backoff:
    base: float = 1.0
    factor: float = 2.0
    cap: float = 60.0
    max_attempts: int = 6

    def delay(self, attempt: int, rng: random.Random, retry_after: float | None = None) -> float:
        """Full-jitter delay before retry number ``attempt`` (1-based)."""
        ceiling = min(self.cap, self.base * self.factor ** (attempt - 1))
        jittered = rng.uniform(0.0, ceiling)
        if retry_after is not None:
            return min(self.cap, max(jittered, retry_after))
        return jittered


class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Opens after ``threshold`` consecutive failures; after ``cooldown`` seconds one
    half-open probe is allowed: success closes the breaker, failure re-opens it."""

    def __init__(self, threshold: int, cooldown: float, clock: Clock) -> None:
        self.threshold = threshold
        self.cooldown = cooldown
        self.clock = clock
        self.failures = 0
        self.state = BreakerState.CLOSED
        self.opened_at = 0.0
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            if self.state == BreakerState.CLOSED:
                return True
            if self.state == BreakerState.OPEN:
                if self.clock.time() - self.opened_at >= self.cooldown:
                    self.state = BreakerState.HALF_OPEN
                    return True
                return False
            return False  # half-open: a probe is already in flight

    def record_success(self) -> None:
        with self._lock:
            self.failures = 0
            self.state = BreakerState.CLOSED

    def record_failure(self) -> None:
        with self._lock:
            self.failures += 1
            if self.state == BreakerState.HALF_OPEN or self.failures >= self.threshold:
                self.state = BreakerState.OPEN
                self.opened_at = self.clock.time()


def call_with_deadline(fn: Callable[[], T], timeout: float) -> T:
    """Run ``fn`` in a daemon thread; raise :class:`DeadlineExceededError` after ``timeout``.

    Libraries such as deep-translator call ``requests`` without a timeout, so a hung
    socket would block forever. The worker is abandoned on timeout; being a daemon thread
    it can never keep the process alive at exit (a ThreadPoolExecutor worker would).
    """
    box: dict[str, object] = {}

    def runner() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller's thread
            box["error"] = exc

    worker = threading.Thread(target=runner, name="mangaar-deadline", daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise DeadlineExceededError(f"call exceeded its {timeout:.1f}s deadline")
    if "error" in box:
        raise box["error"]  # type: ignore[misc]
    return box["value"]  # type: ignore[return-value]


class ResilientCaller:
    """Limiter + breaker + deadline + retries with backoff for one provider."""

    def __init__(
        self,
        name: str,
        *,
        limiter: TokenBucket,
        breaker: CircuitBreaker,
        backoff: Backoff,
        deadline: float,
        clock: Clock,
        rng: random.Random | None = None,
    ) -> None:
        self.name = name
        self.limiter = limiter
        self.breaker = breaker
        self.backoff = backoff
        self.deadline = deadline
        self.clock = clock
        self.rng = rng or random.Random(0)
        self.calls = 0

    def call(self, fn: Callable[[], T]) -> T:
        last: ProviderError | None = None
        for attempt in range(1, self.backoff.max_attempts + 1):
            if not self.breaker.allow():
                raise CircuitOpenError(f"{self.name}: circuit open")
            self.limiter.acquire()
            self.calls += 1
            try:
                result = call_with_deadline(fn, self.deadline)
            except ProviderError as exc:
                self.breaker.record_failure()
                last = exc
                if not exc.retryable or attempt == self.backoff.max_attempts:
                    break
                delay = self.backoff.delay(attempt, self.rng, exc.retry_after)
                log.info("%s: %s; retry %d in %.1fs", self.name, exc, attempt, delay)
                self.clock.sleep(delay)
                continue
            self.breaker.record_success()
            return result
        assert last is not None
        raise last
