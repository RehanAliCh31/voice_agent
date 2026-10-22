"""Normalisation rules, tested against transcripts Whisper actually produced."""

import pytest

from voicebot.asr.normalizer import normalize
from voicebot.dialogue.slots import parse_amount, parse_card_last4

# --- real Whisper output -----------------------------------------------------
# Left column is what Whisper returned for audio of the right column's meaning,
# captured from openai/whisper-base.en. These are observations, not guesses.


@pytest.mark.parametrize(
    "whisper_said,expected",
    [
        ("Transfer $50 to Sarah from Checking.", "transfer 50 dollars to sarah from checking"),
        ("Freeze the card, ending 4-4-1-7.", "freeze the card ending 4417"),
        (
            "What is the balance on my savings account?",
            "what is the balance on my savings account",
        ),
        (
            "For Transfer 50, dollars to Sarah from Checking.",
            "for transfer 50 dollars to sarah from checking",
        ),
    ],
)
def test_observed_transcripts_normalise_as_expected(whisper_said, expected):
    assert normalize(whisper_said) == expected


# --- currency ----------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("$50", "50 dollars"),
        ("$ 50", "50 dollars"),
        ("£20", "20 pounds"),
        ("€35", "35 euros"),
        ("$1,000", "1000 dollars"),
        ("$12.50", "12 50 dollars"),
    ],
)
def test_currency_symbols_become_spoken_words(text, expected):
    assert normalize(text) == expected


def test_a_normalised_amount_still_parses():
    """The whole point: the dialogue manager must be able to read it."""
    assert parse_amount(normalize("Transfer $50 to Sarah.")) == (5000, "USD")
    assert parse_amount(normalize("send £20")) == (2000, "GBP")


# --- digit runs --------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("4-4-1-7", "4417"),
        ("4 4 1 7", "4417"),
        ("card ending 9-0-2-1", "card ending 9021"),
    ],
)
def test_dictated_digits_are_joined(text, expected):
    assert normalize(text) == expected


def test_a_normalised_card_number_still_parses():
    assert parse_card_last4(normalize("Freeze the card, ending 4-4-1-7.")) == "4417"


def test_a_two_digit_run_is_left_alone():
    """ "4-4" is more likely a date or a score than a card number."""
    assert normalize("4-4") == "4 4"


def test_an_ordinary_number_is_untouched():
    assert normalize("transfer 250 dollars") == "transfer 250 dollars"


# --- fillers and punctuation -------------------------------------------------


def test_hesitations_are_dropped():
    assert normalize("um, what is uh my balance") == "what is my balance"


def test_a_word_containing_a_filler_is_kept():
    assert "mummy" in normalize("send it to mummy")


def test_punctuation_is_stripped_but_apostrophes_survive():
    assert normalize("What's my balance?") == "what's my balance"


def test_case_is_folded():
    assert normalize("FREEZE My CARD") == "freeze my card"


# --- edges -------------------------------------------------------------------


def test_empty_input_returns_empty():
    assert normalize("") == ""
    assert normalize("   ") == ""


def test_output_never_has_repeated_spaces():
    assert "  " not in normalize("transfer   $50    to   sarah")
