"""The numeric guard: no spoken figure the database did not supply.

The guard checks money and only money. A bot that cannot say "five working
days" without tripping an alarm is not holding a conversation, so the scope is
deliberately narrow — and inside that scope it is absolute.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from voicebot.agent.guard import guard, money_in_results, money_in_text

FALLBACK = "I have that, but let me read it back from your account."


# --- what counts as money ----------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Your balance is $19,020.00.", [Decimal("19020.00")]),
        ("Your balance is $19,020.", [Decimal("19020")]),
        ("You spent 2483.50 last month.", [Decimal("2483.50")]),
        ("That's 50 dollars.", [Decimal("50")]),
        ("$5 and $10.25", [Decimal("5"), Decimal("10.25")]),
    ],
)
def test_money_is_recognised(text: str, expected: list[Decimal]) -> None:
    assert money_in_text(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "It should arrive in five working days.",
        "The card ending 4417 is now frozen.",
        "I can help with 5 things.",
        "Hello, how can I help?",
        "A replacement arrives in 5 working days.",
    ],
)
def test_plain_numbers_are_not_money(text: str) -> None:
    assert money_in_text(text) == []


def test_money_is_read_out_of_tool_results() -> None:
    results = [
        {"balance": "$19,020.00", "balance_cents": 1_902_000},
        {"cards": [{"last4": "4417", "frozen": True}]},
    ]
    # Both spellings of the same figure collapse to one value; 4417 is not money.
    assert money_in_results(results) == {Decimal("19020.00")}


def test_nested_results_are_searched() -> None:
    results = [{"accounts": [{"balance_cents": 248_350}, {"balance_cents": 1_902_000}]}]
    assert money_in_results(results) == {Decimal("2483.50"), Decimal("19020.00")}


# --- the guard ---------------------------------------------------------------


def test_a_figure_that_matches_a_tool_result_passes() -> None:
    results = [{"balance": "$19,020.00", "balance_cents": 1_902_000}]
    text, hit = guard("Your savings balance is $19,020.00.", results, FALLBACK)
    assert hit is False
    assert text == "Your savings balance is $19,020.00."


def test_a_reformatted_but_correct_figure_passes() -> None:
    # The real model drops a trailing .00. That is a formatting choice, not a
    # different number, so the comparison is by value and not by string.
    results = [{"balance": "$19,020.00"}]
    text, hit = guard("Your balance is $19,020.", results, FALLBACK)
    assert hit is False
    assert text == "Your balance is $19,020."


def test_an_invented_figure_is_caught_and_replaced() -> None:
    results = [{"balance": "$19,020.00"}]
    text, hit = guard("Your balance is $31,415.00.", results, FALLBACK)
    assert hit is True
    assert text == FALLBACK


def test_a_derived_total_is_caught() -> None:
    # Arithmetic is forbidden: $21,503.50 is correct but no tool returned it,
    # and a guard that admits correct-looking sums admits incorrect ones too.
    results = [{"balance": "$19,020.00"}, {"balance": "$2,483.50"}]
    _, hit = guard("Together that's $21,503.50.", results, FALLBACK)
    assert hit is True


def test_one_wrong_figure_among_right_ones_is_still_caught() -> None:
    results = [{"a": "$19,020.00"}, {"b": "$2,483.50"}]
    _, hit = guard("Savings is $19,020.00 and checking is $9,999.99.", results, FALLBACK)
    assert hit is True


@pytest.mark.parametrize(
    "sentence",
    [
        "It should arrive in five working days.",
        "The card ending 4417 is now frozen.",
        "Hello, how can I help?",
    ],
)
def test_non_money_sentences_pass_untouched(sentence: str) -> None:
    text, hit = guard(sentence, [], FALLBACK)
    assert hit is False
    assert text == sentence


def test_a_money_figure_on_a_turn_with_no_tool_calls_is_caught() -> None:
    # Nothing was looked up, so the bot has no business stating a figure.
    text, hit = guard("Your balance is $500.00.", [], FALLBACK)
    assert hit is True
    assert text == FALLBACK


def test_an_empty_reply_is_left_to_the_caller() -> None:
    text, hit = guard("", [{"balance": "$1.00"}], FALLBACK)
    assert hit is False
    assert text == ""
