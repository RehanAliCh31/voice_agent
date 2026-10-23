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

# Spoken replies spell amounts out, so they have to be read rather than
# refused. Written here rather than pulled from a library: the vocabulary a
# bank assistant uses is this list, and a dependency for forty words is a
# dependency to keep up to date for forty words.
_UNITS: dict[str, int] = {
    "zero": 0,
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_SCALES: dict[str, int] = {"hundred": 100, "thousand": 1_000, "million": 1_000_000}

_NUMBER_WORDS = "|".join(sorted(set(_UNITS) | set(_SCALES), key=len, reverse=True))

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

# A figure written in words next to a currency word, with its cents in either
# of the two forms people use out loud: "and ninety-nine cents", or just
# "ninety-nine" straight after the dollars. The bare form is the commoner one
# in speech and used to be read as no cents at all.
_SPELLED = re.compile(
    rf"\b(?P<whole>(?:{_NUMBER_WORDS})(?:[\s-]+(?:and[\s-]+)?(?:{_NUMBER_WORDS}))*)"
    rf"\s+(?:{_USD_WORDS})\b"
    rf"(?:\s+and\s+(?P<cents>(?:{_NUMBER_WORDS})(?:[\s-]+(?:{_NUMBER_WORDS}))*)\s+cents?\b"
    rf"|\s+(?P<bare>(?:{_NUMBER_WORDS})(?:[\s-]+(?:{_NUMBER_WORDS}))*)\b)?",
    re.IGNORECASE,
)

# "a" and "an" are number words to this parser, which makes the "an" inside
# "and" look like one cent. The trailing boundary above stops that; this stops
# the standalone article in "ten dollars a piece" doing the same.
_ARTICLES = frozenset({"a", "an"})

# Any amount in a currency the bank does not hold.
_FOREIGN = re.compile(
    rf"(?: [£€] \s* \d[\d,]*(?:\.\d+)? )"
    rf"| (?: \d[\d,]*(?:\.\d+)? \s* (?:{_OTHER_CURRENCY})\b )"
    rf"| (?: \b(?:{_OTHER_CURRENCY})\s+\d[\d,]*(?:\.\d+)? )",
    re.VERBOSE | re.IGNORECASE,
)


# Two or more number words in a row, with no currency word to mark them:
# "seven sixty", "twenty three forty". One word alone is deliberately not a
# run -- "twenty" is a date or a count far more often than it is $20.00, and
# there is nothing in the sentence to tell them apart.
_WORD_RUN = re.compile(
    rf"\b(?:{_NUMBER_WORDS})(?:[\s-]+(?:{_NUMBER_WORDS}))+\b",
    re.IGNORECASE,
)


def words_to_number(words: str) -> int | None:
    """Read an English number phrase. Returns None if it is not one.

    Scales multiply what precedes them and accumulate: "nineteen thousand
    twenty" is 19,000 then 20, and "eight hundred twenty" is 800 then 20.
    """
    tokens = [w for w in re.split(r"[\s-]+", words.lower().strip()) if w and w != "and"]
    if not tokens:
        return None

    total = 0
    current = 0
    seen = False
    for token in tokens:
        if token in _UNITS:
            current += _UNITS[token]
            seen = True
        elif token in _SCALES:
            scale = _SCALES[token]
            if scale == 100:
                current = max(current, 1) * scale
            else:
                total += max(current, 1) * scale
                current = 0
            seen = True
        else:
            return None
    return total + current if seen else None


def _to_decimal(raw: str) -> Decimal | None:
    try:
        return Decimal(raw.replace(",", ""))
    except InvalidOperation:
        return None


def money_in_text(text: str) -> list[Decimal]:
    """Every money figure the sentence states, in digits or in words."""
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
    for match in _SPELLED.finditer(text or ""):
        parsed = _spelled_amount(match)
        if parsed is not None:
            found.append(parsed[0])
    return found


def _spelled_amount(match: re.Match[str]) -> tuple[Decimal, str] | None:
    """What one spelled match states, plus any trailing words it declined.

    "ten dollars ninety-nine" is how a person says $10.99, so a bare number
    after the currency word is read as the cents -- but only when it could be
    cents. In "two dollars one hundred" the hundred is something else, and it
    is handed back rather than swallowed, because the rewrite below replaces
    everything the pattern matched.

    Both the check and the rewrite go through here, so the figure the guard
    verifies and the figure it prints can never be different figures.
    """
    whole = words_to_number(match.group("whole"))
    if whole is None:
        return None
    amount = Decimal(whole)

    explicit = match.group("cents")
    if explicit:
        cents = words_to_number(explicit)
        return None if cents is None else (amount + Decimal(cents) / 100, "")

    bare = match.group("bare")
    if bare:
        cents = None if bare.strip().lower() in _ARTICLES else words_to_number(bare)
        if cents is not None and 1 <= cents <= 99:
            return amount + Decimal(cents) / 100, ""
        # Matched, but not as cents -- so it is handed straight back. Anything
        # the pattern consumed and the rewrite does not use would otherwise
        # vanish from the sentence.
        return amount, bare
    return amount, ""


def unverifiable_figure(text: str) -> str | None:
    """A figure that cannot be checked at all. Returns what was found, or None.

    Only foreign currencies qualify. This bank holds one, so an amount in
    another was converted -- which the prompt forbids and no tool could have
    produced. Amounts written in words are read and checked like any other.
    """
    foreign = _FOREIGN.search(text or "")
    return foreign.group(0) if foreign else None


def _spoken_amount(run: str, allowed: set[int]) -> int | None:
    """The cents a bare spoken run means, when the bank reported exactly that.

    "seven sixty" is $7.60 to a person and two numbers to a parser, and the
    sentence around it does not settle which. So the sentence is not asked:
    every reading the words could carry is worked out, and one is accepted
    only if the bank produced that figure this turn. A run that means nothing
    the bank reported is not an amount, whatever it resembles.
    """
    words = [w for w in re.split(r"[\s-]+", run.strip()) if w]

    readings: list[int] = []
    whole = words_to_number(" ".join(words))
    if whole is not None:
        readings.append(whole * 100)
    for split in range(1, len(words)):
        dollars = words_to_number(" ".join(words[:split]))
        cents = words_to_number(" ".join(words[split:]))
        if dollars is not None and cents is not None and 0 <= cents <= 99:
            readings.append(dollars * 100 + cents)

    return next((value for value in readings if value in allowed), None)


def in_digits(text: str, sourced_cents: Iterable[int] = ()) -> str:
    """Rewrite amounts written in words as figures: "$3,000.00".

    The model writes them the way a person says them, because its replies are
    read aloud. On screen that is harder to read and harder to check at a
    glance, which matters when the figure is the reader's own money. The same
    parser that verifies a spelled-out amount can rewrite it, so the two can
    never disagree.

    Two passes, because amounts arrive in two shapes. One says "dollars" and
    needs nothing else to identify it. The other drops the word entirely --
    "Starbucks, seven sixty" -- and is only an amount if it happens to be one
    of the figures the bank reported this turn. `sourced_cents` is what
    settles that, so the rewrite never has to guess from the sentence.

    Speech synthesisers read a currency figure back as words, so the spoken
    reply is unchanged by this.
    """

    def replace(match: re.Match[str]) -> str:
        parsed = _spelled_amount(match)
        if parsed is None:
            return match.group(0)
        amount, trailing = parsed
        figure = f"${amount:,.2f}"
        return f"{figure} {trailing}" if trailing else figure

    written = _SPELLED.sub(replace, text or "")

    allowed = set(sourced_cents)
    if not allowed:
        return written

    def replace_run(match: re.Match[str]) -> str:
        cents = _spoken_amount(match.group(0), allowed)
        return match.group(0) if cents is None else f"${Decimal(cents) / 100:,.2f}"

    return _WORD_RUN.sub(replace_run, written)


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
