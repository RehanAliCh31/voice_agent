"""Deadlines, retries and circuit breakers: how a call to a provider is made.

The caller is waiting in silence for every second spent here, so the rules
are strict:

* Every call has a deadline. A retry is only made if it can finish inside
  what is left of it; otherwise the error goes up at once, to a fallback.
* Only retryable errors are retried (see voicebot.errors), with exponential
  backoff and full jitter, a few times at most. A Retry-After from the
  provider is honoured -- and when it is longer than the time left, nothing
  is retried at all.
* A provider that keeps failing is skipped for a while. Its circuit breaker
  opens after a few failures in a row, so the next calls fail in
  microseconds rather than each waiting out a timeout, and after a cooldown
  one call is let through to see whether it has recovered.

Nothing here retries an action with side effects. The callers are the model,
the recognisers and the voices -- reads, every one -- and the bank's writes
never come through this module.
"""

from __future__ import annotations

import contextvars
import random
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar

from voicebot.errors import Cancelled, ErrorCode, ProviderUnavailable, VoicebotError
from voicebot.utils.logging import get_logger

log = get_logger("utils.resilience")

T = TypeVar("T")


class Deadline:
    """A point in time a piece of work must finish by."""

    def __init__(self, seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self.seconds = seconds
        self.at = clock() + seconds

    def remaining(self) -> float:
        return max(0.0, self.at - self._clock())

    @property
    def expired(self) -> bool:
        return self.remaining() <= 0.0

    def cap(self, timeout: float) -> float:
        """`timeout`, or less if the deadline comes first."""
        return min(timeout, self.remaining())


# The deadline of the turn in hand, and whether its caller is still there.
# Set once at the top of a turn; every provider call below reads them, so
# no signature has to carry them through.
_DEADLINE: contextvars.ContextVar[Deadline | None] = contextvars.ContextVar(
    "deadline", default=None
)


def current_deadline() -> Deadline | None:
    return _DEADLINE.get()


@contextmanager
def deadline_context(seconds: float) -> Iterator[Deadline]:
    """Everything inside shares one deadline, `seconds` from now."""
    deadline = Deadline(seconds)
    token = _DEADLINE.set(deadline)
    try:
        yield deadline
    finally:
        _DEADLINE.reset(token)


class CancelToken:
    """Set when the caller has gone -- closed the page, hung up, started over.

    Work checks it at the points where stopping leaves nothing half done:
    before the recogniser, before each model round, before the voice.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self.reason = ""

    def cancel(self, reason: str = "the caller left") -> None:
        self.reason = reason
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def check(self, stage: str) -> None:
        if self._event.is_set():
            raise Cancelled(f"{self.reason} before {stage}", component=stage)


_CANCEL: contextvars.ContextVar[CancelToken | None] = contextvars.ContextVar("cancel", default=None)


@contextmanager
def cancel_context(token: CancelToken) -> Iterator[CancelToken]:
    reset = _CANCEL.set(token)
    try:
        yield token
    finally:
        _CANCEL.reset(reset)


def checkpoint(stage: str) -> None:
    """Stop here if the caller has gone. Does nothing outside a turn."""
    token = _CANCEL.get()
    if token is not None:
        token.check(stage)


@dataclass(frozen=True)
class RetryPolicy:
    """How many times to try, and how long to wait between tries.

    `attempts` counts the first try, so 2 means one retry.
    """

    attempts: int = 2
    base_s: float = 0.2
    cap_s: float = 1.0
    # A retry that cannot have at least this long to run is not worth making.
    min_attempt_s: float = 0.5

    def delay(self, attempt: int, retry_after: float | None = None) -> float:
        """Full jitter: anything up to base * 2**attempt, capped; Retry-After is a floor."""
        backoff = random.uniform(0.0, min(self.cap_s, self.base_s * 2**attempt))
        return max(backoff, retry_after or 0.0)


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Skips a provider that keeps failing, and tries it again after a rest.

    Closed: calls go through. After `failures` retryable failures in a row it
    opens: calls are refused at once. After `cooldown_s` one call is let
    through (half-open); if it succeeds the breaker closes, and if it fails
    it opens again for another cooldown.

    Only failures that say the provider is in trouble count -- timeouts, 5xx,
    429, a dropped connection. A 400 says our request was wrong, and a
    breaker that opened on it would hide the bug behind an outage.
    """

    def __init__(
        self,
        name: str,
        failures: int = 3,
        cooldown_s: float = 20.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = name
        self.failures = failures
        self.cooldown_s = cooldown_s
        self._clock = clock
        self._lock = threading.Lock()
        self._count = 0
        self._opened_at: float | None = None
        self._probing = False
        self.opened_total = 0

    @property
    def state(self) -> BreakerState:
        with self._lock:
            return self._state()

    def _state(self) -> BreakerState:
        if self._opened_at is None:
            return BreakerState.CLOSED
        if self._clock() - self._opened_at >= self.cooldown_s:
            return BreakerState.HALF_OPEN
        return BreakerState.OPEN

    def allow(self) -> bool:
        """May a call go through now? In half-open, only one at a time."""
        with self._lock:
            state = self._state()
            if state is BreakerState.CLOSED:
                return True
            if state is BreakerState.HALF_OPEN and not self._probing:
                self._probing = True
                return True
            return False

    def success(self) -> None:
        with self._lock:
            if self._opened_at is not None:
                log.info("circuit %s closed: the provider answered again", self.name)
            self._count = 0
            self._opened_at = None
            self._probing = False

    def failure(self) -> bool:
        """Count a failure. True when this one opened the circuit."""
        with self._lock:
            self._count += 1
            was_probing, self._probing = self._probing, False
            if was_probing or self._count >= self.failures:
                opened = self._opened_at is None or was_probing
                self._opened_at = self._clock()
                if opened:
                    self.opened_total += 1
                return opened
            return False

    def release(self) -> None:
        """A probe that ended without a verdict (bad input, say) frees the slot."""
        with self._lock:
            self._probing = False


_BREAKERS: dict[str, CircuitBreaker] = {}
_BREAKERS_LOCK = threading.Lock()
_BREAKER_DEFAULTS = {"failures": 3, "cooldown_s": 20.0}


def configure_breakers(failures: int, cooldown_s: float) -> None:
    """Set the thresholds breakers made from now on will use."""
    with _BREAKERS_LOCK:
        _BREAKER_DEFAULTS.update(failures=failures, cooldown_s=cooldown_s)
        for existing in _BREAKERS.values():
            existing.failures = failures
            existing.cooldown_s = cooldown_s


def breaker(name: str) -> CircuitBreaker:
    """The one breaker for provider `name`, shared by every call to it."""
    with _BREAKERS_LOCK:
        found = _BREAKERS.get(name)
        if found is None:
            found = CircuitBreaker(
                name,
                failures=int(_BREAKER_DEFAULTS["failures"]),
                cooldown_s=float(_BREAKER_DEFAULTS["cooldown_s"]),
            )
            _BREAKERS[name] = found
        return found


def breakers() -> dict[str, CircuitBreaker]:
    with _BREAKERS_LOCK:
        return dict(_BREAKERS)


def reset_breakers() -> None:
    """Forget every breaker. For tests."""
    with _BREAKERS_LOCK:
        _BREAKERS.clear()


def call_with_retry(
    attempt: Callable[[float], T],
    *,
    component: str,
    timeout_s: float,
    policy: RetryPolicy | None = None,
    deadline: Deadline | None = None,
    circuit: CircuitBreaker | None = None,
    sleep: Callable[[float], None] | None = None,
    error_type: type[VoicebotError] = ProviderUnavailable,
) -> T:
    """Call `attempt(timeout)` until it succeeds or there is no point trying again.

    `attempt` gets the timeout to use for that one try, already cut down to
    what the deadline allows, and must raise a VoicebotError when it fails --
    anything else is a bug and goes straight up. An open circuit is raised
    as `error_type`, the provider's own error, so whatever catches that
    provider's failures catches this one too.
    """
    # Imported here: observability imports this module's breakers for metrics.
    from voicebot.utils.observability import report

    policy = policy or RetryPolicy()
    sleep = sleep or time.sleep
    deadline = (
        deadline
        or current_deadline()
        or Deadline(timeout_s * policy.attempts + policy.cap_s * policy.attempts)
    )

    for number in range(policy.attempts):
        if circuit is not None and not circuit.allow():
            raise error_type(
                f"{circuit.name} is failing; its circuit is open",
                component=component,
                code=ErrorCode.PROVIDER_UNAVAILABLE,
                retryable=False,
            )
        budget = deadline.cap(timeout_s)
        if budget < policy.min_attempt_s and number > 0:
            break
        started = time.monotonic()
        try:
            result = attempt(max(budget, policy.min_attempt_s))
        except VoicebotError as exc:
            latency_ms = (time.monotonic() - started) * 1000
            if exc.component == VoicebotError.default_component:
                exc.component = component
            opened = False
            if circuit is not None:
                if exc.retryable:
                    opened = circuit.failure()
                else:
                    circuit.release()
            notes = [f"circuit {circuit.name} opened"] if opened and circuit else []

            pause = policy.delay(number, exc.retry_after)
            if not exc.retryable or opened:
                retrying = False
            elif number == policy.attempts - 1:
                retrying = False
            elif pause + policy.min_attempt_s > deadline.remaining():
                retrying = False
                notes.append("no time left to retry")
            else:
                retrying = True
                notes.append(f"retrying in {pause:.2f}s")

            report(
                exc,
                attempt=number + 1,
                latency_ms=latency_ms,
                note="; ".join(notes) or None,
                alert=opened,
            )
            if not retrying:
                raise
            sleep(pause)
            continue
        if circuit is not None:
            circuit.success()
        return result

    raise error_type(
        f"{component}: out of time before another attempt", component=component, retryable=False
    )
