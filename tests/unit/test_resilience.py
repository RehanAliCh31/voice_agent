"""The error model, retries, deadlines, circuit breakers, and how a failure is reported.

These are the rules every provider call follows, so they are tested on their
own, with a fake clock and a fake sleep: no test here waits for real time.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest

from voicebot.errors import (
    AuthFailure,
    ErrorCode,
    InvalidInput,
    ProviderTimeout,
    ProviderUnavailable,
    RateLimited,
    VoicebotError,
    call_context,
    code_for_status,
    retry_after_seconds,
)
from voicebot.utils import observability
from voicebot.utils.logging import FORMAT, RedactingFormatter, redact
from voicebot.utils.resilience import (
    BreakerState,
    CircuitBreaker,
    Deadline,
    RetryPolicy,
    breaker,
    call_with_retry,
    reset_breakers,
)


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture(autouse=True)
def _fresh() -> Iterator[None]:
    reset_breakers()
    observability.reset_counts()
    yield
    reset_breakers()
    observability.reset_counts()


@pytest.fixture
def logged() -> Iterator[list[logging.LogRecord]]:
    """Every record the voicebot loggers write, with the real formatter's call id."""
    records: list[logging.LogRecord] = []

    class Keep(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = Keep()
    root = logging.getLogger("voicebot")
    root.addHandler(handler)
    yield records
    root.removeHandler(handler)


# --- the error model ---------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (408, ErrorCode.PROVIDER_TIMEOUT),
        (504, ErrorCode.PROVIDER_TIMEOUT),
        (429, ErrorCode.RATE_LIMITED),
        (401, ErrorCode.AUTH_FAILURE),
        (403, ErrorCode.AUTH_FAILURE),
        (500, ErrorCode.PROVIDER_UNAVAILABLE),
        (503, ErrorCode.PROVIDER_UNAVAILABLE),
        (400, ErrorCode.INVALID_INPUT),
        (422, ErrorCode.INVALID_INPUT),
    ],
)
def test_an_http_status_names_its_kind_of_failure(status: int, code: ErrorCode) -> None:
    assert code_for_status(status) is code


def test_only_temporary_failures_are_retryable() -> None:
    assert ProviderTimeout("x").retryable
    assert ProviderUnavailable("x").retryable
    assert RateLimited("x").retryable
    assert not InvalidInput("x").retryable
    assert not AuthFailure("x").retryable
    # A provider error can carry any code, and its code decides.
    assert not ProviderUnavailable("x", code=ErrorCode.INVALID_INPUT).retryable


def test_an_error_knows_the_call_it_happened_in() -> None:
    with call_context("call-42"):
        error = ProviderTimeout("slow", component="llm", cause=TimeoutError())
    assert error.call_id == "call-42"
    assert error.fields() == {
        "code": "provider_timeout",
        "component": "llm",
        "call": "call-42",
        "retryable": True,
        "cause": "TimeoutError",
    }
    assert isinstance(error.__cause__, TimeoutError)


@pytest.mark.parametrize(
    ("header", "seconds"),
    [("3", 3.0), ("0.5", 0.5), ("-2", 0.0), ("", None), (None, None), ("soon", None)],
)
def test_retry_after_in_seconds(header: str | None, seconds: float | None) -> None:
    assert retry_after_seconds(header) == seconds


def test_retry_after_as_an_http_date() -> None:
    assert retry_after_seconds("Thu, 01 Jan 1970 00:00:10 GMT", now=4.0) == pytest.approx(6.0)


# --- backoff -----------------------------------------------------------------


def test_backoff_is_jittered_and_capped() -> None:
    policy = RetryPolicy(attempts=5, base_s=0.2, cap_s=1.0)
    for attempt in range(6):
        for _ in range(50):
            assert 0.0 <= policy.delay(attempt) <= min(1.0, 0.2 * 2**attempt)


def test_retry_after_is_a_floor_on_the_wait() -> None:
    assert RetryPolicy().delay(0, retry_after=1.5) >= 1.5


# --- circuit breakers ----------------------------------------------------------


def test_a_breaker_opens_after_failures_in_a_row_and_refuses_at_once() -> None:
    clock = Clock()
    circuit = CircuitBreaker("llm", failures=3, cooldown_s=20, clock=clock)
    assert [circuit.failure() for _ in range(3)] == [False, False, True]
    assert circuit.state is BreakerState.OPEN
    assert not circuit.allow()


def test_one_success_resets_the_count() -> None:
    circuit = CircuitBreaker("llm", failures=3, clock=Clock())
    circuit.failure()
    circuit.failure()
    circuit.success()
    circuit.failure()
    assert circuit.state is BreakerState.CLOSED


def test_after_the_cooldown_one_call_probes_the_provider() -> None:
    clock = Clock()
    circuit = CircuitBreaker("tts", failures=1, cooldown_s=20, clock=clock)
    circuit.failure()
    clock.now += 20
    assert circuit.state is BreakerState.HALF_OPEN
    assert circuit.allow()
    assert not circuit.allow(), "only one probe at a time"
    circuit.success()
    assert circuit.state is BreakerState.CLOSED


def test_a_failed_probe_opens_it_again_for_a_whole_cooldown() -> None:
    clock = Clock()
    circuit = CircuitBreaker("tts", failures=3, cooldown_s=20, clock=clock)
    for _ in range(3):
        circuit.failure()
    clock.now += 20
    assert circuit.allow()
    assert circuit.failure() is True
    clock.now += 19
    assert circuit.state is BreakerState.OPEN


def test_breakers_are_one_per_provider() -> None:
    assert breaker("asr-en") is breaker("asr-en")
    assert breaker("asr-en") is not breaker("asr-ur")


# --- retrying --------------------------------------------------------------------


def _flaky(*errors: VoicebotError, result: str = "ok"):  # type: ignore[no-untyped-def]
    """An attempt that raises each of `errors` in turn, then succeeds."""
    calls: list[float] = []
    queue = list(errors)

    def attempt(timeout: float) -> str:
        calls.append(timeout)
        if queue:
            raise queue.pop(0)
        return result

    return attempt, calls


def test_a_temporary_failure_is_retried_and_succeeds() -> None:
    attempt, calls = _flaky(ProviderUnavailable("500", component="llm"))
    slept: list[float] = []
    out = call_with_retry(
        attempt, component="llm", timeout_s=5, policy=RetryPolicy(attempts=2), sleep=slept.append
    )
    assert out == "ok" and len(calls) == 2 and len(slept) == 1


def test_a_permanent_failure_is_never_retried() -> None:
    attempt, calls = _flaky(InvalidInput("400", component="llm"))
    with pytest.raises(InvalidInput):
        call_with_retry(attempt, component="llm", timeout_s=5, policy=RetryPolicy(attempts=3))
    assert len(calls) == 1


def test_attempts_are_bounded() -> None:
    attempt, calls = _flaky(*[ProviderTimeout("slow") for _ in range(10)])
    with pytest.raises(ProviderTimeout):
        call_with_retry(
            attempt,
            component="asr",
            timeout_s=1,
            policy=RetryPolicy(attempts=3),
            sleep=lambda _: None,
        )
    assert len(calls) == 3


def test_each_attempt_gets_only_what_the_deadline_has_left() -> None:
    clock = Clock()
    attempt, calls = _flaky()
    call_with_retry(attempt, component="llm", timeout_s=10, deadline=Deadline(4, clock=clock))
    assert calls == [4]


def test_no_retry_when_the_wait_would_outlast_the_deadline() -> None:
    clock = Clock()
    attempt, calls = _flaky(RateLimited("429", retry_after=30))
    with pytest.raises(RateLimited):
        call_with_retry(
            attempt,
            component="llm",
            timeout_s=5,
            policy=RetryPolicy(attempts=3),
            deadline=Deadline(5, clock=clock),
            sleep=lambda _: pytest.fail("must not wait 30 s with the caller listening"),
        )
    assert len(calls) == 1


def test_retry_after_is_honoured_when_there_is_time() -> None:
    attempt, _ = _flaky(RateLimited("429", retry_after=0.75))
    slept: list[float] = []
    call_with_retry(
        attempt,
        component="llm",
        timeout_s=5,
        policy=RetryPolicy(attempts=2),
        deadline=Deadline(10),
        sleep=slept.append,
    )
    assert slept and slept[0] >= 0.75


def test_an_open_circuit_fails_fast_without_calling() -> None:
    circuit = CircuitBreaker("llm", failures=1, cooldown_s=60, clock=Clock())
    circuit.failure()
    attempt, calls = _flaky()
    with pytest.raises(ProviderUnavailable, match="circuit is open") as raised:
        call_with_retry(attempt, component="llm", timeout_s=5, circuit=circuit)
    assert calls == [] and not raised.value.retryable


def test_the_failure_that_opens_a_circuit_is_not_retried_and_alerts(
    logged: list[logging.LogRecord],
) -> None:
    circuit = CircuitBreaker("llm", failures=1, cooldown_s=60, clock=Clock())
    attempt, calls = _flaky(ProviderUnavailable("500"), ProviderUnavailable("500"))
    with pytest.raises(ProviderUnavailable):
        call_with_retry(
            attempt, component="llm", timeout_s=5, circuit=circuit, sleep=lambda _: None
        )
    assert len(calls) == 1
    assert any(r.levelno == logging.CRITICAL and "ALERT" in r.getMessage() for r in logged)
    assert observability.event_counts()["alerts"] == 1


def test_bad_input_does_not_open_a_circuit() -> None:
    circuit = CircuitBreaker("llm", failures=1, clock=Clock())
    attempt, _ = _flaky(InvalidInput("400"))
    with pytest.raises(InvalidInput):
        call_with_retry(attempt, component="llm", timeout_s=5, circuit=circuit)
    assert circuit.state is BreakerState.CLOSED


# --- reporting ---------------------------------------------------------------------


def test_every_failure_is_logged_with_its_call_and_counted(
    logged: list[logging.LogRecord],
) -> None:
    attempt, _ = _flaky(ProviderTimeout("slow", component="asr"), ProviderTimeout("slow"))
    with call_context("call-7"), pytest.raises(ProviderTimeout):
        call_with_retry(attempt, component="asr", timeout_s=1, sleep=lambda _: None)

    lines = [r.getMessage() for r in logged if r.name == "voicebot.errors"]
    assert len(lines) == 2
    assert all("call=call-7" in line and "code=provider_timeout" in line for line in lines)
    assert "attempt=1" in lines[0] and "retrying" in lines[0]
    assert observability.error_counts()[("asr", "provider_timeout")] == 2


def test_alerts_are_rate_limited(logged: list[logging.LogRecord]) -> None:
    assert observability.send_alert("llm:down", "the model is down")
    assert not observability.send_alert("llm:down", "the model is down")
    assert observability.send_alert("tts:down", "the voice is down")


def test_an_alert_goes_to_the_webhook_when_one_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[tuple[str, dict]] = []
    monkeypatch.setenv(observability.ALERT_ENV, "http://alerts.test/hook")
    monkeypatch.setattr(observability, "_post_alert", lambda url, body: sent.append((url, body)))

    class Inline:
        def __init__(self, target, args, **_):  # type: ignore[no-untyped-def]
            self.target, self.args = target, args

        def start(self) -> None:
            self.target(*self.args)

    monkeypatch.setattr(observability.threading, "Thread", Inline)
    observability.send_alert("db:down", "mail ops@example.com", call="c1")
    assert sent[0][0] == "http://alerts.test/hook"
    assert sent[0][1]["message"] == "mail <email>" and sent[0][1]["call"] == "c1"


def test_metrics_show_errors_and_open_circuits() -> None:
    circuit = breaker("tts-ur")
    for _ in range(circuit.failures):
        circuit.failure()
    observability.report(ProviderUnavailable("down", component="tts-ur"))
    text = "\n".join(observability.metrics_lines())
    assert 'voicebot_errors_total{component="tts-ur",code="provider_unavailable"} 1' in text
    assert 'voicebot_circuit_open{provider="tts-ur"} 1' in text


# --- logging -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "masked"),
    [
        ("write to ali.khan@example.com", "write to <email>"),
        ("call +92 300 1234567", "call <number …4567>"),
        ("or 0300-1234567", "or <number …4567>"),
        ("card 4417 1234 5678 9012", "card <number …9012>"),
        ("Authorization: Bearer abc.def", "Authorization: Bearer <redacted>"),
        ("confirmation_token=q1w2e3", "confirmation_token=<redacted>"),
        ("pin=1234", "pin=<redacted>"),
    ],
)
def test_personal_details_and_secrets_are_masked(text: str, masked: str) -> None:
    assert redact(text) == masked


@pytest.mark.parametrize(
    "kept",
    [
        "statement for 2026-10-29 17:56:00",
        "balance $19,020.00 in 4513 ms",
        "pin changed on card 4417",
        "session 096b0ab6aec442bc90ccd5b6d8b5acdb",
    ],
)
def test_ordinary_lines_are_left_alone(kept: str) -> None:
    assert redact(kept) == kept


def test_a_line_names_its_call_and_masks_only_the_message() -> None:
    record = logging.LogRecord(
        "voicebot.x", logging.INFO, __file__, 1, "reach me at %s", ("a@b.co",), None
    )
    record.call_id = "call-9"
    line = RedactingFormatter(FORMAT).format(record)
    assert "[call=call-9] | reach me at <email>" in line
    assert record.args == ("a@b.co",), "the record itself is left as it was"
