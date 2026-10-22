"""The last check before the bot speaks: every figure came from the database.

v1 could not invent a balance, because every sentence it could say was a
template. v2 writes its own sentences, so that property has to be enforced
rather than assumed. This module is where it is enforced.

**The allowed set comes from the bank, never from text.** An earlier version
built it by re-parsing the JSON of each tool result, which was defeated by
laundering: calling `get_balance(account="$31,415.00")` makes the bank's error
message echo that string back, and a guard that reads result text then treats
$31,415.00 as sourced. The model now gets no say in what counts as sourced —
the executor reports the integer cents it took from the database, and only
those are trusted.

**What counts as a figure.** A money-shaped number in digits, a figure spelled
out in words next to a currency word, or any amount in a currency this bank
does not hold. Plain numbers are left alone: "five working days", "the card
ending 4417" and "I can help with four things" are not claims about money, and
a guard that stops those is not protecting anyone.

**Arithmetic is a violation**, including correct arithmetic. "Together that's
$21,503.50" is suppressed even when the sum is right, because a rule that
admits correct-looking totals admits incorrect ones by the same rule. The cost
is a bot that cannot add up out loud.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from decimal import Decimal, InvalidOperation
from typing import Any

from voicebot.utils.logging import get_logger

log = get_logger("agent.guard")

# The one currency this bank holds. Anything else was converted, which the
# prompt forbids and no tool could have produced.
_USD_WORDS = r"dollars?|usd|bucks?|cents?"
_OTHER_CURRENCY = r"pounds?|sterling|gbp|euros?|eur|rupees?|pkr|yen|jpy"

_NUMBER_WORDS = (
    r"zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million"
)

# Digits that are money: a currency symbol either side, a currency word after,
# or two decimal places.
_MONEY = re.compile(
    rf"""
    (?: [$£€] \s* (?P<symbol>\d[\d,]*(?:\.\d+)?) )
    | (?: (?P<worded>\d[\d,]*(?:\.\d+)?) \s* (?:{_USD_WORDS})\b )
    | (?P<decimal>\b\d[\d,]*\.\d{{2}}\b)
    """,
    re.VERBOSE | re.IGNORECASE,
)

# A currency word in front of a bare number: "USD 31415".
_PREFIXED = re.compile(rf"\b(?:{_USD_WORDS})\s+(?P<amount>\d[\d,]*(?:\.\d+)?)\b", re.IGNORECASE)

# A figure written in words next to a currency word. The value cannot be
# recovered reliably, and does not need to be: the prompt tells the model to
# write figures in digits, so words are a violation by policy.
_SPELLED = re.compile(
    rf"\b(?:{_NUMBER_WORDS})(?:[\s-]+(?:and[\s-]+)?(?:{_NUMBER_WORDS}))*\s+"
    rf"(?:{_USD_WORDS}|{_OTHER_CURRENCY})\b",
    re.IGNORECASE,
)

# Any amount in a currency the bank does not hold.
_FOREIGN = re.compile(
    rf"(?: [£€] \s* \d[\d,]*(?:\.\d+)? )"
    rf"| (?: \d[\d,]*(?:\.\d+)? \s* (?:{_OTHER_CURRENCY})\b )"
    rf"| (?: \b(?:{_OTHER_CURRENCY})\s+\d[\d,]*(?:\.\d+)? )",
    re.VERBOSE | re.IGNORECASE,
)


def _to_decimal(raw: str) -> Decimal | None:
    try:
        return Decimal(raw.replace(",", ""))
    except InvalidOperation:
        return None


def money_in_text(text: str) -> list[Decimal]:
    """Every money figure the sentence states in digits, in order."""
    found: list[Decimal] = []
    for match in _MONEY.finditer(text or ""):
        raw = match.group("symbol") or match.group("worded") or match.group("decimal")
        value = _to_decimal(raw)
        if value is not None:
            found.append(value)
    for match in _PREFIXED.finditer(text or ""):
        value = _to_decimal(match.group("amount"))
        if value is not None:
            found.append(value)
    return found


def unverifiable_figure(text: str) -> str | None:
    """A figure that cannot be checked at all. Returns what was found, or None.

    Spelled-out amounts and foreign currencies are violations outright: the
    first because its value cannot be recovered, the second because this bank
    holds one currency, so any other one was converted.
    """
    spelled = _SPELLED.search(text or "")
    if spelled:
        return spelled.group(0)
    foreign = _FOREIGN.search(text or "")
    if foreign:
        return foreign.group(0)
    return None


def allowed_values(sourced_cents: Iterable[int]) -> set[Decimal]:
    """The figures the bank produced, as the guard compares them."""
    return {Decimal(cents) / 100 for cents in sourced_cents}


def fallback_sentence(sourced_cents: Sequence[int]) -> str:
    """What to say when the model's own sentence is rejected.

    Assembled from values the bank returned, never generated, and never from
    anything the model supplied. This is the one sentence that has to be
    trustworthy by construction.
    """
    amounts = sorted(allowed_values(sourced_cents))
    if not amounts:
        return "Let me check that properly — could you ask me again?"
    if len(amounts) == 1:
        return f"The figure I have is ${amounts[0]:,.2f}."
    readable = ", ".join(f"${amount:,.2f}" for amount in amounts)
    return f"The figures I have are {readable}."


def guard(
    text: str,
    results: Sequence[Any],
    fallback: str,
    sourced_cents: Sequence[int] = (),
) -> tuple[str, bool]:
    """Check the reply before it is spoken.

    Returns the text to say and whether the guard fired. `results` is accepted
    for logging only — it is deliberately **not** used to decide what is
    allowed. On a violation the model's sentence is discarded entirely rather
    than edited: a sentence with one invented figure removed is still a
    sentence built on an invention.
    """
    if not text:
        return text, False

    unverifiable = unverifiable_figure(text)
    if unverifiable:
        log.warning("guard suppressed an unverifiable figure %r in %r", unverifiable, text)
        return fallback, True

    spoken = money_in_text(text)
    if not spoken:
        return text, False

    allowed = allowed_values(sourced_cents)
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
