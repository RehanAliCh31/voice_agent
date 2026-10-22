"""Generate a templated banking corpus with exact BIO slot tags.

CLINC150 has intent labels but no slot annotations, so the slot head has no
real supervision. We generate it: because the generator knows exactly where
each value was inserted, the BIO tags are correct by construction rather than
by annotation.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

SLOT_NAMES: tuple[str, ...] = (
    "amount",
    "payee",
    "date",
    "time_period",
    "account_type",
    "card_last4",
    "merchant",
    "bill_type",
)

# Multi-word values are deliberate: they exercise the B- to I- transition.
VALUES: dict[str, tuple[str, ...]] = {
    "amount": (
        "fifty dollars",
        "twenty pounds",
        "a hundred dollars",
        "250 dollars",
        "thirty five euros",
        "1000 rupees",
        "five hundred dollars",
    ),
    "payee": (
        "sarah",
        "john smith",
        "my landlord",
        "ahmed",
        "the electric company",
        "mum",
        "david lee",
    ),
    "date": (
        "today",
        "tomorrow",
        "next friday",
        "the first of march",
        "monday morning",
        "the 15th",
    ),
    "time_period": (
        "last month",
        "this week",
        "the last three months",
        "yesterday",
        "this year",
    ),
    "account_type": (
        "checking",
        "savings",
        "my current account",
        "the joint account",
    ),
    "card_last4": ("4417", "9021", "3356", "7788"),
    "merchant": (
        "amazon",
        "the corner shop",
        "uber",
        "starbucks",
        "tesco",
        "netflix",
    ),
    "bill_type": ("electricity", "water", "phone", "credit card", "internet"),
}

TEMPLATES: dict[str, tuple[str, ...]] = {
    "transfer": (
        "transfer {amount} to {payee}",
        "send {amount} to {payee} on {date}",
        "move {amount} from {account_type} to {payee}",
        "{payee} needs {amount} by {date}",
        "please transfer {amount} to {payee} from {account_type}",
    ),
    "pay_bill": (
        "pay my {bill_type} bill",
        "pay {amount} towards my {bill_type} bill",
        "settle the {bill_type} bill on {date}",
        "i want to pay {amount} for {bill_type}",
    ),
    "spending_history": (
        "how much did i spend at {merchant} {time_period}",
        "show my spending {time_period}",
        "what did i pay {merchant} {time_period}",
        "list transactions from {account_type} {time_period}",
    ),
    "balance": (
        "what is the balance on {account_type}",
        "how much is in {account_type}",
        "balance for {account_type} please",
    ),
    "freeze_account": (
        "freeze the card ending {card_last4}",
        "block card {card_last4}",
        "stop all payments on {account_type}",
    ),
    "report_fraud": (
        "i did not make the {amount} payment to {merchant}",
        "there is a fraudulent charge from {merchant} {time_period}",
        "report the card ending {card_last4} as stolen",
    ),
    "new_card": (
        "order a new card for {account_type}",
        "replace the card ending {card_last4}",
        "i need a new card by {date}",
    ),
}

_PLACEHOLDER = re.compile(r"\{(\w+)\}")


@dataclass(frozen=True)
class SlotExample:
    """One synthetic utterance with intent and token-aligned BIO tags."""

    text: str
    intent: str
    tokens: list[str]
    tags: list[str]


def slot_tag_vocab() -> list[str]:
    """The BIO tag vocabulary: O first, then B-/I- for every slot, sorted."""
    tags = ["O"]
    for name in sorted(SLOT_NAMES):
        tags.extend([f"B-{name}", f"I-{name}"])
    return tags


def _render_from(
    intent: str, template: str, values: dict[str, tuple[str, ...]], rng: random.Random
) -> SlotExample:
    """Render one example drawing slot values from the given pool.

    re.split with a capturing group alternates literal, capture, literal, ...
    so odd indices are slot names. Using position rather than a membership
    test keeps this correct even if literal text ever equals a slot name.
    """
    tokens: list[str] = []
    tags: list[str] = []
    for index, part in enumerate(_PLACEHOLDER.split(template)):
        if not part:
            continue
        if index % 2 == 1:
            value_tokens = rng.choice(values[part]).split()
            tokens.extend(value_tokens)
            tags.append(f"B-{part}")
            tags.extend([f"I-{part}"] * (len(value_tokens) - 1))
        else:
            words = part.split()
            tokens.extend(words)
            tags.extend(["O"] * len(words))
    return SlotExample(text=" ".join(tokens), intent=intent, tokens=tokens, tags=tags)


def _render(intent: str, template: str, rng: random.Random) -> SlotExample:
    """Render using the full value pool."""
    return _render_from(intent, template, VALUES, rng)


def generate(n: int, seed: int = 42) -> list[SlotExample]:
    """Generate `n` examples, cycling templates so every intent is represented."""
    rng = random.Random(seed)
    pairs = [(intent, tpl) for intent, templates in TEMPLATES.items() for tpl in templates]
    examples = [_render(*pairs[i % len(pairs)], rng) for i in range(n)]
    rng.shuffle(examples)
    return examples


def generate_split(
    n_train: int,
    n_eval: int,
    seed: int = 42,
    eval_templates_per_intent: int = 1,
    eval_value_fraction: float = 0.3,
) -> tuple[list[SlotExample], list[SlotExample]]:
    """Split by TEMPLATE and by VALUE, never by example.

    Splitting a generated corpus by example leaks: every phrasing appears on
    both sides, so the evaluation measures memorisation of a fixed value list
    and scores a perfect 1.0. Holding out whole templates AND a slice of each
    value pool makes the evaluation measure the two things that matter in
    deployment -- finding a slot in a phrasing the model has never seen, and
    recognising a value it has never seen.

    Returns (train, evaluation).
    """
    rng = random.Random(seed)

    train_templates: list[tuple[str, str]] = []
    eval_templates: list[tuple[str, str]] = []
    for intent, templates in TEMPLATES.items():
        shuffled = list(templates)
        rng.shuffle(shuffled)
        held = min(eval_templates_per_intent, max(0, len(shuffled) - 1))
        eval_templates.extend((intent, t) for t in shuffled[:held])
        train_templates.extend((intent, t) for t in shuffled[held:])

    train_values: dict[str, tuple[str, ...]] = {}
    eval_values: dict[str, tuple[str, ...]] = {}
    for slot, pool in VALUES.items():
        shuffled = list(pool)
        rng.shuffle(shuffled)
        held = min(max(1, int(len(shuffled) * eval_value_fraction)), len(shuffled) - 1)
        eval_values[slot] = tuple(shuffled[:held])
        train_values[slot] = tuple(shuffled[held:])

    train = [
        _render_from(*train_templates[i % len(train_templates)], train_values, rng)
        for i in range(n_train)
    ]
    evaluation = [
        _render_from(*eval_templates[i % len(eval_templates)], eval_values, rng)
        for i in range(n_eval)
    ]
    rng.shuffle(train)
    rng.shuffle(evaluation)
    return train, evaluation
