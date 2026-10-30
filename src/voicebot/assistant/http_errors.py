"""What an httpx failure, or an HTTP status, means -- for the recognisers and voices."""

from __future__ import annotations

import httpx

from voicebot.errors import ErrorCode, VoicebotError, code_for_status, retry_after_seconds
from voicebot.utils.logging import get_logger

log = get_logger("assistant.http")


def from_transport(
    exc: httpx.HTTPError, what: str, error_type: type[VoicebotError], timeout: float
) -> VoicebotError:
    """A request that never got an answer: timed out, refused, reset, cut off."""
    if isinstance(exc, httpx.TimeoutException):
        return error_type(
            f"{what} did not answer within {timeout:.1f}s",
            code=ErrorCode.PROVIDER_TIMEOUT,
            cause=exc,
        )
    return error_type(f"{what} could not be reached: {type(exc).__name__}", cause=exc)


def from_status(
    response: httpx.Response,
    what: str,
    error_type: type[VoicebotError],
    message: str | None = None,
) -> VoicebotError:
    """An answer with an error status. Its body goes to DEBUG only: it can hold speech.

    `message` replaces the usual "<what> answered <status>", for a caller
    whose logs have long said it another way.
    """
    try:
        body = response.text[:200]
    except httpx.StreamError:  # a streamed answer whose body could not be read
        body = "(not read)"
    log.debug("%s answered %d: %s", what, response.status_code, body)
    return error_type(
        message or f"{what} answered {response.status_code}",
        code=code_for_status(response.status_code),
        retry_after=retry_after_seconds(response.headers.get("Retry-After")),
    )
