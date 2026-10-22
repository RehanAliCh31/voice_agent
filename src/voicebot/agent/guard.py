"""The last check before the bot speaks: every figure came from the database.

v1 could not invent a balance, because every sentence it could say was a
template. v2 writes its own sentences, so that property has to be enforced
rather than assumed. This module is where it is enforced.

**It checks money, and only money.** A figure counts when it carries a
currency symbol, is followed by a currency word, or is written with exactly
two decimal places. "Five working days", "the card ending 4417" and "I can
help with three things" pass untouched — a guard that stops those is not
protecting anyone, it is preventing a conversation.

Inside that scope it is absolute, including arithmetic. "Together that's
$21,503.50" is caught even when the sum is right, because a guard that admits
correct-looking totals admits incorrect ones by the same rule. The cost is
that the bot cannot add up out loud; the benefit is that it cannot be wrong
about money.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation
from typing import Any

from voicebot.utils.logging import get_logger

log = get_logger("agent.guard")

_CURRENCY_WORDS = r"dollars?|usd|cents?|bucks?"

# A number is money when a currency symbol leads it, a currency word follows
# it, or it is written with exactly two decimal places -- the three ways an
# amount actually appears in a spoken banking reply.
_MONEY = re.compile(
    rf"""
    (?: [$£€] \s* (?P<symbol>\d[\d,]*(?:\.\d+)?) )
    | (?: (?P<worded>\d[\d,]*(?:\.\d+)?) \s* (?:{_CURRENCY_WORDS})\b )
    | (?P<decimal>\b\d[\d,]*\.\d{{2}}\b)
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _to_decimal(raw: str) -> Decimal | None:
    try:
        return Decimal(raw.replace(",", ""))
    except InvalidOperation:
        return None


def money_in_text(text: str) -> list[Decimal]:
    """Every money figure the sentence states, in order."""
    found: list[Decimal] = []
    for match in _MONEY.finditer(text or ""):
        raw = match.group("symbol") or match.group("worded") or match.group("decimal")
        value = _to_decimal(raw)
        if value is not None:
            found.append(value)
    return found


def money_in_results(results: Sequence[Any]) -> set[Decimal]:
    """Every money figure the tools returned, however deeply nested.

    A `*_cents` integer is read as cents; a money-shaped string is read as
    written. Both spellings of one amount collapse to the same Decimal, which
    is what lets `$19,020` match `$19,020.00`.
    """
    found: set[Decimal] = set()

    def walk(node: Any, key: str = "") -> None:
        if isinstance(node, dict):
            for name, value in node.items():
                walk(value, name)
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value, key)
        elif isinstance(node, bool):
            return  # bool is an int in Python; it is never an amount
        elif isinstance(node, int) and key.endswith("_cents"):
            found.add(Decimal(node) / 100)
        elif isinstance(node, str):
            found.update(money_in_text(node))

    walk(list(results))
    return found


def guard(text: str, results: Sequence[Any], fallback: str) -> tuple[str, bool]:
    """Check the reply before it is spoken.

    Returns the text to say and whether the guard fired. On a violation the
    model's sentence is discarded entirely rather than edited: a sentence with
    one invented figure removed is still a sentence built on an invention.
    """
    if not text:
        return text, False

    spoken = money_in_text(text)
    if not spoken:
        return text, False

    allowed = money_in_results(results)
    unsourced = [value for value in spoken if value not in allowed]
    if not unsourced:
        return text, False

    log.warning(
        "guard suppressed a reply: %s not in %s | text=%r",
        [str(v) for v in unsourced],
        sorted(str(v) for v in allowed),
        text,
    )
    return fallback, True
