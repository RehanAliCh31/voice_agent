"""A single place that configures logging, so modules never call basicConfig.

Every line names the call it belongs to -- `[call=<id>]`, the conversation's
session id, or `-` outside one -- so one failed call can be followed from
start to end with a single grep. And every message is masked on its way out:
email addresses, phone and card numbers, and tokens are replaced before the
line is written, wherever in the code it was logged from.

VOICEBOT_LOG_LEVEL sets the level (INFO unless it says otherwise). What a
customer said is logged at DEBUG only.
"""

from __future__ import annotations

import logging
import os
import re
import sys

_HANDLER_NAME = "voicebot-stdout"
FORMAT = "%(asctime)s %(levelname)-7s %(name)s [call=%(call_id)s] | %(message)s"

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# Ten or more digits, spaces and dashes allowed between them: a phone or card
# number. A date and time ("2026-10-29 17") has the same shape and is kept.
_NUMBER = re.compile(r"(?<![\w.])\+?\d(?:[ -]?\d){9,18}(?![\w.])")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}(?: \d{2})?")
_BEARER = re.compile(r"(?i)\b(bearer\s+)[^\s'\",}]+")
_SECRET = re.compile(
    r"(?i)\b(token|confirmation_token|api[_-]?key|password|secret|pin)(['\"]?\s*[=:]\s*['\"]?)"
    r"[^\s'\",}]+"
)


def _mask_number(match: re.Match[str]) -> str:
    found = match.group(0)
    if _DATE.fullmatch(found):
        return found
    digits = re.sub(r"\D", "", found)
    return f"<number …{digits[-4:]}>"


def redact(text: str) -> str:
    """`text` with personal details and secrets masked."""
    text = _EMAIL.sub("<email>", text)
    text = _BEARER.sub(lambda m: f"{m.group(1)}<redacted>", text)
    text = _SECRET.sub(lambda m: f"{m.group(1)}{m.group(2)}<redacted>", text)
    return _NUMBER.sub(_mask_number, text)


class _CallIdFilter(logging.Filter):
    """Stamps each record with the call it belongs to."""

    def filter(self, record: logging.LogRecord) -> bool:
        from voicebot.errors import current_call_id

        if not hasattr(record, "call_id"):
            record.call_id = current_call_id() or "-"
        return True


class RedactingFormatter(logging.Formatter):
    """Masks the message, and only the message: the timestamp is left alone."""

    def format(self, record: logging.LogRecord) -> str:
        original = record.msg, record.args
        try:
            record.msg = redact(record.getMessage())
        except Exception:  # a message that cannot be built is logged as it came
            return super().format(record)
        record.args = None
        try:
            return super().format(record)
        finally:
            record.msg, record.args = original


def _level() -> int:
    name = os.environ.get("VOICEBOT_LOG_LEVEL", "INFO").strip().upper()
    return logging.getLevelNamesMapping().get(name, logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """Return a logger writing to stdout at INFO level, or VOICEBOT_LOG_LEVEL.

    Whether the handler is already attached is read off the logger itself, not
    a module-level flag. `logging` keeps its own registry that survives a
    module reload, so a flag would reset while the old handler stayed -- and
    every reload would add another, printing each line one more time. That is
    not hypothetical: reloading the package in a notebook does it immediately.
    """
    root = logging.getLogger("voicebot")
    if not any(getattr(h, "name", None) == _HANDLER_NAME for h in root.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.name = _HANDLER_NAME
        handler.setFormatter(RedactingFormatter(FORMAT))
        handler.addFilter(_CallIdFilter())
        root.setLevel(_level())
        root.addHandler(handler)
        root.propagate = False
    return logging.getLogger(f"voicebot.{name}")
