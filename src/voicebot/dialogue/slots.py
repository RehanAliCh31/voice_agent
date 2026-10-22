"""What each intent needs before it can be acted on.

The dialogue manager asks for one missing slot at a time, in the order listed
in `required`. `confirm` marks the intents that move money or disable a card:
those are read back to the user before anything happens.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Written out rather than imported from a library: the slot values this bot
# sees come from a known generator, and a dependency for twenty words is not
# worth the install.
_NUMBER_WORDS: dict[str, int] = {
    "zero": 0,
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
_MULTIPLIERS: dict[str, int] = {"hundred": 100, "thousand": 1000}

_CURRENCIES: dict[str, str] = {
    "dollars": "USD",
    "dollar": "USD",
    "pounds": "GBP",
    "pound": "GBP",
    "euros": "EUR",
    "euro": "EUR",
    "rupees": "PKR",
    "rupee": "PKR",
}

AFFIRMATIVE = frozenset(
    {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "confirm", "correct", "right", "please"}
)
NEGATIVE = frozenset({"no", "nope", "nah", "cancel", "stop", "don't", "dont", "negative"})


@dataclass(frozen=True)
class IntentSpec:
    """Slots an intent needs, in the order they should be asked for."""

    required: tuple[str, ...]
    confirm: bool


# Only these intents are actionable. Anything else CLINC150 recognises is
# answered informationally; the bot does not pretend to act on it.
INTENT_SPECS: dict[str, IntentSpec] = {
    "transfer": IntentSpec(required=("amount", "payee"), confirm=True),
    "pay_bill": IntentSpec(required=("bill_type",), confirm=True),
    "balance": IntentSpec(required=("account_type",), confirm=False),
    "spending_history": IntentSpec(required=("time_period",), confirm=False),
    "freeze_account": IntentSpec(required=("card_last4",), confirm=True),
    "report_fraud": IntentSpec(required=("card_last4",), confirm=True),
    "new_card": IntentSpec(required=("account_type",), confirm=True),
}

SLOT_PROMPTS: dict[str, str] = {
    "amount": "How much would you like to transfer?",
    "payee": "Who should I send it to?",
    "date": "Which date should I schedule that for?",
    "time_period": "Over what period? Last month, this week?",
    "account_type": "Which account — checking or savings?",
    "card_last4": "Which card? Tell me the last four digits.",
    "merchant": "Which merchant?",
    "bill_type": "Which bill — electricity, water, phone or internet?",
}


def parse_amount(text: str) -> tuple[int, str] | None:
    """Parse a spoken amount into (cents, currency code).

    Money is handled in integer cents throughout. Floats accumulate rounding
    error, and a transfer that is a hundredth of a penny wrong is a bug nobody
    wants to debug.

    Returns None when no number is present.
    """
    words = re.findall(r"[a-z0-9]+", text.lower())
    currency = "USD"
    for word in words:
        if word in _CURRENCIES:
            currency = _CURRENCIES[word]
            break

    total = 0
    current = 0
    seen_number = False
    for word in words:
        if word.isdigit():
            current += int(word)
            seen_number = True
        elif word in _NUMBER_WORDS:
            current += _NUMBER_WORDS[word]
            seen_number = True
        elif word in _MULTIPLIERS:
            # "a hundred" has no preceding number; treat it as one hundred.
            current = max(current, 1) * _MULTIPLIERS[word]
            seen_number = True
        elif word == "and":
            continue

    if not seen_number:
        return None
    total += current
    return total * 100, currency


def parse_card_last4(text: str) -> str | None:
    """Pull a four-digit card suffix out of a slot value."""
    digits = re.findall(r"\d", text)
    if len(digits) < 4:
        return None
    return "".join(digits[-4:])


def is_affirmative(text: str) -> bool:
    """True when the user agreed. Checked before `is_negative`."""
    words = set(re.findall(r"[a-z']+", text.lower()))
    return bool(words & AFFIRMATIVE) and not (words & NEGATIVE)


def is_negative(text: str) -> bool:
    """True when the user declined."""
    words = set(re.findall(r"[a-z']+", text.lower()))
    return bool(words & NEGATIVE)
