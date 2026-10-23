"""The numeric guard: no spoken figure the database did not supply.

The guard checks money and only money. A bot that cannot say "five working
days" without tripping an alarm is not holding a conversation, so the scope is
deliberately narrow -- and inside that scope it is absolute.

What it trusts is the integer cents the executor reports taking from the bank.
It does not read the tool results' text: an earlier version did, and could be
defeated by calling a tool with a money-shaped argument that the error message
echoed back.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from voicebot.agent.guard import fallback_sentence, guard, in_digits, money_in_text

FALLBACK = "I have that, but let me read it back from your account."
SAVINGS = 1_902_000  # $19,020.00
CHECKING = 248_350  # $2,483.50


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
    ],
)
def test_plain_numbers_are_not_money(text: str) -> None:
    assert money_in_text(text) == []


# --- the guard ---------------------------------------------------------------


def test_a_figure_that_matches_a_sourced_value_passes() -> None:
    text, hit = guard("Your savings balance is $19,020.00.", [], FALLBACK, (SAVINGS,))
    assert hit is False
    assert text == "Your savings balance is $19,020.00."


def test_a_reformatted_but_correct_figure_passes() -> None:
    # The real model drops a trailing .00. That is a formatting choice, not a
    # different number, so the comparison is by value and not by string.
    _, hit = guard("Your balance is $19,020.", [], FALLBACK, (SAVINGS,))
    assert hit is False


def test_an_invented_figure_is_caught_and_replaced() -> None:
    text, hit = guard("Your balance is $31,415.00.", [], FALLBACK, (SAVINGS,))
    assert hit is True
    assert text == FALLBACK


def test_a_derived_total_is_caught() -> None:
    # Arithmetic is forbidden: $21,503.50 is correct but no tool returned it,
    # and a guard that admits correct-looking sums admits incorrect ones too.
    _, hit = guard("Together that's $21,503.50.", [], FALLBACK, (SAVINGS, CHECKING))
    assert hit is True


def test_one_wrong_figure_among_right_ones_is_still_caught() -> None:
    _, hit = guard(
        "Savings is $19,020.00 and checking is $9,999.99.", [], FALLBACK, (SAVINGS, CHECKING)
    )
    assert hit is True


def test_a_money_figure_on_a_turn_with_no_tool_calls_is_caught() -> None:
    # Nothing was looked up, so the bot has no business stating a figure.
    text, hit = guard("Your balance is $500.00.", [], FALLBACK, ())
    assert hit is True
    assert text == FALLBACK


def test_an_empty_reply_is_left_to_the_caller() -> None:
    text, hit = guard("", [], FALLBACK, (SAVINGS,))
    assert hit is False
    assert text == ""


# --- the allowed set comes from the bank, never from rendered text ----------


def test_a_figure_laundered_through_an_error_message_is_caught() -> None:
    """The guard must not read money out of text the model itself supplied.

    Attack: call get_balance(account="$31,415.00"). The bank's error echoes
    the argument back, and a guard that re-parses result text then treats
    $31,415.00 as sourced. Verified end to end before this test existed.
    """
    results = [
        {
            "error": "I don't recognise the account '$31,415.00'. I have: checking, savings.",
            "hint": "I don't recognise the account '$31,415.00'.",
        }
    ]
    _, hit = guard("Your savings balance is $31,415.00.", results, FALLBACK, sourced_cents=())
    assert hit is True


def test_a_figure_laundered_through_a_confirmation_summary_is_caught() -> None:
    results = [
        {
            "status": "needs_confirmation",
            "token": "abc",
            "summary": "Transfer $0.01 to my $77,000.00 fund from checking",
        }
    ]
    _, hit = guard("You have $77,000.00 available.", results, FALLBACK, sourced_cents=(1,))
    assert hit is True


def test_sourced_cents_are_what_the_guard_trusts() -> None:
    # The same sentence passes only when the bank actually produced the figure.
    _, hit = guard("Your balance is $19,020.00.", [], FALLBACK, sourced_cents=(1_902_000,))
    assert hit is False


# --- figures the model writes in words, or in another currency --------------


@pytest.mark.parametrize(
    "sentence",
    [
        "Your savings balance is about nineteen thousand dollars.",
        "Your savings balance is forty two thousand dollars.",
        "You have around five hundred dollars left.",
    ],
)
def test_a_spelled_out_amount_is_caught(sentence: str) -> None:
    # The prompt tells the model to speak like a person, which invites exactly
    # this. A figure in words is unverifiable, so it is a violation outright.
    _, hit = guard(sentence, [], FALLBACK, sourced_cents=(1_902_000,))
    assert hit is True


def test_a_currency_word_before_the_number_is_caught() -> None:
    _, hit = guard("Your balance is USD 31415.", [], FALLBACK, sourced_cents=(1_902_000,))
    assert hit is True


@pytest.mark.parametrize(
    "sentence", ["You have 500 pounds.", "That's £500 left.", "About 300 euros."]
)
def test_a_non_usd_figure_is_always_a_violation(sentence: str) -> None:
    # The bank holds one currency. A figure in another one was converted,
    # which the prompt forbids and no tool could have returned.
    _, hit = guard(sentence, [], FALLBACK, sourced_cents=(50_000,))
    assert hit is True


@pytest.mark.parametrize(
    "sentence",
    [
        "It should arrive in five working days.",
        "The card ending 4417 is now frozen.",
        "I can help with four things.",
        "Hello, how can I help?",
    ],
)
def test_ordinary_sentences_still_pass(sentence: str) -> None:
    text, hit = guard(sentence, [], FALLBACK, sourced_cents=())
    assert hit is False
    assert text == sentence


def test_the_fallback_cannot_speak_a_laundered_figure() -> None:
    # The fallback is built from sourced cents, so a figure the guard just
    # rejected cannot come back in the sentence meant to replace it.

    assert "31,415" not in fallback_sentence((1_902_000,))
    assert "19,020.00" in fallback_sentence((1_902_000,))


# --- figures the model writes in words ---------------------------------------
#
# Replies are spoken aloud, so the model writes amounts the way a person says
# them. An earlier version treated any spelled-out figure as unverifiable and
# suppressed it, which silenced entirely correct answers -- "savings is at
# nineteen thousand twenty dollars" against a real balance of $19,020.00.
# They are parsed and checked like any other figure now.


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("nineteen thousand twenty dollars", [Decimal("19020")]),
        ("two hundred dollars", [Decimal("200")]),
        ("eighteen thousand eight hundred twenty dollars", [Decimal("18820")]),
        ("two thousand four hundred and eighty three dollars", [Decimal("2483")]),
        ("fifty dollars", [Decimal("50")]),
        ("zero dollars", [Decimal("0")]),
        ("a hundred dollars", [Decimal("100")]),
        ("one million dollars", [Decimal("1000000")]),
        ("fifty dollars and fifty cents", [Decimal("50.50")]),
        ("two hundred dollars and five cents", [Decimal("200.05")]),
    ],
)
def test_amounts_written_in_words_are_read(text: str, expected: list[Decimal]) -> None:
    assert money_in_text(text) == expected


def test_a_correct_spelled_out_figure_passes() -> None:
    text, hit = guard("Savings is at nineteen thousand twenty dollars.", [], FALLBACK, (SAVINGS,))
    assert hit is False
    assert text.startswith("Savings")


def test_a_wrong_spelled_out_figure_is_caught() -> None:
    _, hit = guard("Savings is at forty two thousand dollars.", [], FALLBACK, (SAVINGS,))
    assert hit is True


def test_the_reported_sentence_now_passes() -> None:
    # Verbatim from the failure report, with the balances it was reported against.
    _, hit = guard(
        "You've got two accounts. Checking is at zero, and savings is at "
        "nineteen thousand twenty dollars.",
        [],
        FALLBACK,
        (0, SAVINGS),
    )
    assert hit is False


def test_the_reported_transfer_sentence_now_passes() -> None:
    _, hit = guard(
        "Done. Two hundred dollars is on its way to Sara, and your savings now "
        "shows eighteen thousand eight hundred twenty dollars.",
        [],
        FALLBACK,
        (20_000, 1_882_000),
    )
    assert hit is False


@pytest.mark.parametrize(
    "sentence",
    [
        "It should arrive in five working days.",
        "You have two accounts.",
        "The card ending 4417 is now frozen.",
        "I can help with four things.",
    ],
)
def test_number_words_without_a_currency_are_not_figures(sentence: str) -> None:
    # "two accounts" and "five working days" are not amounts.
    assert money_in_text(sentence) == []
    _, hit = guard(sentence, [], FALLBACK, ())
    assert hit is False


# --- showing amounts in digits ----------------------------------------------
#
# The model writes amounts the way a person says them, because its replies are
# read aloud. On screen that is harder to read than "$3,000.00", and harder to
# check at a glance -- which matters when the figure is your money. The same
# parser that verifies a spelled-out amount can rewrite it.


@pytest.mark.parametrize(
    ("spoken", "shown"),
    [
        (
            "Your checking account holds three thousand dollars.",
            "Your checking account holds $3,000.00.",
        ),
        ("That's two hundred dollars to Sara.", "That's $200.00 to Sara."),
        (
            "Savings is at nineteen thousand twenty dollars.",
            "Savings is at $19,020.00.",
        ),
        ("fifty dollars and fifty cents", "$50.50"),
        ("Checking is at zero dollars.", "Checking is at $0.00."),
    ],
)
def test_amounts_in_words_are_rewritten_as_digits(spoken: str, shown: str) -> None:
    assert in_digits(spoken) == shown


@pytest.mark.parametrize(
    "text",
    [
        "Your balance is $19,020.00.",
        "It should arrive in five working days.",
        "You have two accounts.",
        "The card ending 4417 is now frozen.",
        "Hello, how can I help?",
        "",
    ],
)
def test_text_without_a_spelled_out_amount_is_untouched(text: str) -> None:
    assert in_digits(text) == text


def test_every_amount_in_a_sentence_is_rewritten() -> None:
    assert (
        in_digits("Checking is at three thousand dollars and savings at two hundred dollars.")
        == "Checking is at $3,000.00 and savings at $200.00."
    )


# --- dollars and cents, said the way people say them -------------------------
#
# Found live on a transaction query: the model wrote "ten dollars ninety-nine",
# the parser read ten dollars and dropped the rest, and a correct answer about
# the customer's own Netflix charge was suppressed as an invention.


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ten dollars ninety-nine", [Decimal("10.99")]),
        ("two hundred fifteen dollars fifty", [Decimal("215.50")]),
        ("twenty four dollars ninety nine", [Decimal("24.99")]),
        # The explicit form still works, and still wins where both could read.
        ("ten dollars and ninety-nine cents", [Decimal("10.99")]),
        # No trailing number: unchanged.
        ("forty dollars", [Decimal("40")]),
    ],
)
def test_dollars_and_cents_without_the_word_cents(text: str, expected: list[Decimal]) -> None:
    assert money_in_text(text) == expected


def test_a_trailing_number_too_large_to_be_cents_is_not_cents() -> None:
    """ "two dollars one hundred" is not $2.100. Read the dollars, leave the rest."""
    assert money_in_text("two dollars one hundred") == [Decimal("2")]


def test_the_suppressed_netflix_sentence_now_passes() -> None:
    said = "You paid Netflix ten dollars ninety-nine on October nineteenth, from checking."
    text, hit = guard(said, [], FALLBACK, sourced_cents=(1_099,))

    assert hit is False
    assert "Netflix" in text


def test_a_wrong_cents_figure_is_still_caught() -> None:
    _, hit = guard("You paid Netflix ten dollars fifty.", [], FALLBACK, sourced_cents=(1_099,))
    assert hit is True


def test_the_words_are_rewritten_as_one_figure() -> None:
    assert in_digits("You paid ten dollars ninety-nine.") == "You paid $10.99."


def test_words_that_are_not_cents_survive_the_rewrite() -> None:
    """The rewrite replaces whatever it matched, so what it declines to use
    has to come back out intact rather than being swallowed."""
    assert in_digits("two dollars one hundred") == "$2.00 one hundred"


def test_the_an_inside_and_is_not_one_cent() -> None:
    """ "three thousand dollars and savings..." must not become $3,000.01."""
    said = "Checking is at three thousand dollars and savings at two hundred dollars."
    assert money_in_text(said) == [Decimal("3000"), Decimal("200")]


def test_an_article_after_the_amount_is_not_cents() -> None:
    assert money_in_text("ten dollars a piece") == [Decimal("10")]
    assert in_digits("ten dollars a piece") == "$10.00 a piece"


# --- amounts said without a currency word ------------------------------------
#
# In a list the model drops the word "dollars" entirely: "Starbucks, seven
# sixty. Uber, twenty three forty." Those read fine aloud and badly on screen,
# and the old rewrite could not touch them -- out of context "seven sixty" is
# as much a time or a date as an amount.
#
# So context is not what decides. The bank reported certain figures this turn,
# and a spoken run is rewritten only when it means one of them. Nothing else
# is touched, whatever it looks like.


@pytest.mark.parametrize(
    ("said", "sourced", "expected"),
    [
        ("Starbucks, seven sixty.", (760,), "Starbucks, $7.60."),
        ("Uber, twenty three forty.", (2_340,), "Uber, $23.40."),
        ("Netflix, ten ninety nine.", (1_099,), "Netflix, $10.99."),
        ("The card bill is two hundred fifteen.", (21_500,), "The card bill is $215.00."),
    ],
)
def test_a_sourced_spoken_amount_becomes_a_figure(
    said: str, sourced: tuple[int, ...], expected: str
) -> None:
    assert in_digits(said, sourced) == expected


def test_an_unsourced_run_is_left_exactly_alone() -> None:
    """The bank said nothing of the sort, so this is not an amount."""
    said = "It arrives in five working days."
    assert in_digits(said, (760,)) == said


def test_a_date_is_not_an_amount() -> None:
    said = "Netflix, $10.99, on the twenty second."
    assert in_digits(said, (1_099, 2_000)) == said


def test_a_single_word_is_never_read_as_an_amount() -> None:
    """ "twenty" alone is a date or a count far more often than $20.00."""
    assert in_digits("on the twenty second", (2_000,)) == "on the twenty second"


def test_the_currency_word_form_still_needs_no_sourcing() -> None:
    """It says "dollars". There is nothing to disambiguate."""
    assert in_digits("You paid ten dollars ninety-nine.") == "You paid $10.99."
