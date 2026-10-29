"""What a failure leaves behind: one log line, a counter, and sometimes an alert.

Every error is reported the same way, so one failed call can be
reconstructed from the logs alone: grep its call id and each line says which
component failed, with which code, on which attempt, after how long.

Alerts go to the log at CRITICAL, and -- if VOICEBOT_ALERT_WEBHOOK names a
URL -- are POSTed there as JSON, at most once per kind of event per
ALERT_EVERY_S, from a background thread with a short timeout. Nothing in a
call ever waits for an alert to be delivered.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.request
from collections import Counter
from typing import Any

from voicebot.errors import VoicebotError, current_call_id
from voicebot.utils.logging import get_logger, redact

log = get_logger("errors")

ALERT_ENV = "VOICEBOT_ALERT_WEBHOOK"
ALERT_EVERY_S = 300.0
ALERT_TIMEOUT_S = 2.0

_LOCK = threading.Lock()
_ERRORS: Counter[tuple[str, str]] = Counter()
_EVENTS: Counter[str] = Counter()
_LAST_ALERT: dict[str, float] = {}


def report(
    error: VoicebotError,
    *,
    attempt: int | None = None,
    latency_ms: float | None = None,
    note: str | None = None,
    alert: bool = False,
    level: int | None = None,
) -> None:
    """Log `error` as one structured line, count it, and alert if asked."""
    # An error made outside the call -- a client built at startup, say --
    # still belongs to the call that met it.
    if error.call_id is None:
        error.call_id = current_call_id()
    fields = error.fields()
    with _LOCK:
        _ERRORS[(fields["component"], fields["code"])] += 1

    parts = [f"{name}={value}" for name, value in fields.items() if value is not None]
    if attempt is not None:
        parts.append(f"attempt={attempt}")
    if latency_ms is not None:
        parts.append(f"latency_ms={latency_ms:.0f}")
    if level is None:
        level = logging.WARNING if error.retryable else logging.ERROR
    log.log(level, "error %s | %s%s", " ".join(parts), error, f" ({note})" if note else "")

    if alert:
        send_alert(
            f"{fields['component']}:{fields['code']}",
            f"{error} ({note})" if note else str(error),
            **fields,
        )


def count(event: str, amount: int = 1) -> None:
    """Count something worth watching that is not an exception."""
    with _LOCK:
        _EVENTS[event] += amount


def error_counts() -> dict[tuple[str, str], int]:
    with _LOCK:
        return dict(_ERRORS)


def event_counts() -> dict[str, int]:
    with _LOCK:
        return dict(_EVENTS)


def reset_counts() -> None:
    """Forget every count. For tests."""
    with _LOCK:
        _ERRORS.clear()
        _EVENTS.clear()
        _LAST_ALERT.clear()


def send_alert(key: str, message: str, **fields: Any) -> bool:
    """Raise an alert for `key`, unless one was raised for it recently.

    Returns whether it was raised. Logged at CRITICAL either way it goes, and
    sent to the webhook only when one is configured.
    """
    now = time.monotonic()
    with _LOCK:
        last = _LAST_ALERT.get(key)
        if last is not None and now - last < ALERT_EVERY_S:
            return False
        _LAST_ALERT[key] = now
        _EVENTS["alerts"] += 1

    log.critical("ALERT %s | %s", key, message)
    url = os.environ.get(ALERT_ENV, "").strip()
    if url:
        body = {
            "alert": key,
            "message": redact(message),
            "call": fields.get("call") or current_call_id(),
            "at": time.time(),
            **{k: v for k, v in fields.items() if k != "call"},
        }
        threading.Thread(target=_post_alert, args=(url, body), daemon=True, name="alert").start()
    return True


def _post_alert(url: str, body: dict[str, Any]) -> None:
    request = urllib.request.Request(
        url,
        data=json.dumps(body, default=str).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=ALERT_TIMEOUT_S):
            pass
    except Exception as exc:  # an alert that fails must never take anything with it
        log.warning("the alert webhook could not be reached: %s", type(exc).__name__)


def metrics_lines() -> list[str]:
    """Prometheus lines for errors, events and circuit breakers."""
    from voicebot.utils.resilience import BreakerState, breakers

    lines = [
        "# HELP voicebot_errors_total Failures by component and error code.",
        "# TYPE voicebot_errors_total counter",
    ]
    for (component, code), value in sorted(error_counts().items()):
        lines.append(f'voicebot_errors_total{{component="{component}",code="{code}"}} {value}')
    lines += [
        "# HELP voicebot_events_total Things worth watching: escalations, dead letters, alerts.",
        "# TYPE voicebot_events_total counter",
    ]
    for event, value in sorted(event_counts().items()):
        lines.append(f'voicebot_events_total{{event="{event}"}} {value}')
    lines += [
        "# HELP voicebot_circuit_open 1 while a provider's circuit breaker is open.",
        "# TYPE voicebot_circuit_open gauge",
    ]
    for name, found in sorted(breakers().items()):
        is_open = 0 if found.state is BreakerState.CLOSED else 1
        lines.append(f'voicebot_circuit_open{{provider="{name}"}} {is_open}')
    return lines
