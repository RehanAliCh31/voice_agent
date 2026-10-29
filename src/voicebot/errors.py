"""One error model for everything that can fail.

Every failure the bot meets is one of a handful of kinds, and the kind is what
decides what happens next: whether to try again, whether a failing provider
should be skipped for a while, and what the customer is told. So each error
carries its kind as a code, the component it came from, the call it happened
in, and the exception that caused it -- enough to reconstruct a failed call
from the logs alone.

The provider-specific exceptions that existed before this module
(`LLMUnavailable`, `ASRUnavailable`, the assistant's `AsrError`, `TtsError`
and `LlmError`, the bank's `StorageError`) are subclasses of these, so every
`except` already written keeps working.

Retryable means the same request might succeed in a moment: a timeout, a 429,
a 5xx, a dropped connection. A 4xx, bad input or a refused credential will
fail again identically, and is never retried.
"""

from __future__ import annotations

import contextvars
import email.utils
import time
from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum
from typing import Any, ClassVar

# The call a log line or an error belongs to. In the web bot a call is one
# conversation, named by its session id; the hands-free assistant names each
# conversation it opens. Set at the edge, read everywhere below it.
CALL_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar("call_id", default=None)


def current_call_id() -> str | None:
    return CALL_ID.get()


@contextmanager
def call_context(call_id: str | None) -> Iterator[None]:
    """Everything logged or raised inside belongs to `call_id`."""
    token = CALL_ID.set(call_id)
    try:
        yield
    finally:
        CALL_ID.reset(token)


class ErrorCode(StrEnum):
    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    RATE_LIMITED = "rate_limited"
    INVALID_INPUT = "invalid_input"
    AUTH_FAILURE = "auth_failure"
    DATABASE_ERROR = "database_error"
    TOOL_ERROR = "tool_error"
    AUDIO_ERROR = "audio_error"
    SESSION_ERROR = "session_error"
    CANCELLED = "cancelled"
    INTERNAL = "internal"


RETRYABLE: frozenset[ErrorCode] = frozenset(
    {ErrorCode.PROVIDER_TIMEOUT, ErrorCode.PROVIDER_UNAVAILABLE, ErrorCode.RATE_LIMITED}
)


class VoicebotError(Exception):
    """A failure the bot knows how to name.

    The message is for logs and engineers. What the customer hears is chosen
    from `code` (see voicebot.messages) and never contains this text.
    """

    default_code: ClassVar[ErrorCode] = ErrorCode.INTERNAL
    default_component: ClassVar[str] = "unknown"

    def __init__(
        self,
        message: str = "",
        *,
        component: str | None = None,
        code: ErrorCode | None = None,
        call_id: str | None = None,
        cause: BaseException | None = None,
        retry_after: float | None = None,
        retryable: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.code: ErrorCode = code or self.default_code
        self.component: str = component or self.default_component
        self.call_id: str | None = call_id or current_call_id()
        self.cause = cause
        self.retry_after = retry_after
        self._retryable = retryable
        if cause is not None:
            self.__cause__ = cause

    @property
    def retryable(self) -> bool:
        return self.code in RETRYABLE if self._retryable is None else self._retryable

    def fields(self) -> dict[str, Any]:
        """What a structured log line says about this error."""
        return {
            "code": str(self.code),
            "component": self.component,
            "call": self.call_id or "-",
            "retryable": self.retryable,
            "cause": type(self.cause).__name__ if self.cause is not None else None,
        }


class ProviderTimeout(VoicebotError):
    default_code = ErrorCode.PROVIDER_TIMEOUT


class ProviderUnavailable(VoicebotError):
    """A provider could not be reached or answered with a server error.

    Also the base of the provider-specific errors, which may carry any
    provider code -- a timeout, a 429, a refused credential -- in `code`.
    """

    default_code = ErrorCode.PROVIDER_UNAVAILABLE


class RateLimited(VoicebotError):
    default_code = ErrorCode.RATE_LIMITED


class InvalidInput(VoicebotError):
    default_code = ErrorCode.INVALID_INPUT


class AuthFailure(VoicebotError):
    default_code = ErrorCode.AUTH_FAILURE


class DatabaseError(VoicebotError):
    default_code = ErrorCode.DATABASE_ERROR
    default_component = "database"


class ToolError(VoicebotError):
    default_code = ErrorCode.TOOL_ERROR
    default_component = "tool"


class AudioError(VoicebotError):
    default_code = ErrorCode.AUDIO_ERROR
    default_component = "audio"


class SessionError(VoicebotError):
    default_code = ErrorCode.SESSION_ERROR
    default_component = "session"


class Cancelled(VoicebotError):
    """The caller left -- hung up, closed the page, started over -- mid-turn."""

    default_code = ErrorCode.CANCELLED
    default_component = "session"


class InternalError(VoicebotError):
    default_code = ErrorCode.INTERNAL
    default_component = "internal"


def code_for_status(status: int) -> ErrorCode:
    """The kind of failure an HTTP status means."""
    if status in (408, 504):
        return ErrorCode.PROVIDER_TIMEOUT
    if status == 429:
        return ErrorCode.RATE_LIMITED
    if status in (401, 403):
        return ErrorCode.AUTH_FAILURE
    if status >= 500:
        return ErrorCode.PROVIDER_UNAVAILABLE
    return ErrorCode.INVALID_INPUT


def retry_after_seconds(value: str | None, now: float | None = None) -> float | None:
    """A Retry-After header as seconds from now: a number, or an HTTP date."""
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when is None:  # pragma: no cover - parsedate_to_datetime raises instead on 3.12
        return None
    return max(0.0, when.timestamp() - (time.time() if now is None else now))
