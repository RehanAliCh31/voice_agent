"""The SQLite bank: balances, the ledger, and the audit log.

Every test opens its own `:memory:` database, so they are isolated from each
other and from any file on disk, and they exercise real SQL rather than a mock.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from voicebot.bank.base import PERIODS, BankError, format_money, period_range
from voicebot.bank.sqlite_bank import SqliteBank


@pytest.fixture
def bank() -> SqliteBank:
    return SqliteBank()


# --- seeded state ------------------------------------------------------------


def test_seeded_balances(bank: SqliteBank) -> None:
    assert bank.balance("checking") == 248_350
    assert bank.balance("savings") == 1_902_000


def test_an_unknown_account_is_refused(bank: SqliteBank) -> None:
    with pytest.raises(BankError):
        bank.balance("offshore")


def test_the_seeded_cards_and_bills_are_listed(bank: SqliteBank) -> None:
    assert {card.last4 for card in bank.cards()} == {"4417", "9021"}
    assert "electricity" in {bill.name for bill in bank.bills()}


# --- transfers ---------------------------------------------------------------


def test_transfer_moves_money_and_writes_a_ledger_row(bank: SqliteBank) -> None:
    remaining = bank.transfer(5_000, "sarah", "checking")
    assert remaining == 243_350
    assert bank.balance("checking") == 243_350
    # The ledger row is dated now, so it counts towards today's spending.
    assert bank.spending("today") == 5_000


def test_transfer_refuses_more_than_the_balance(bank: SqliteBank) -> None:
    with pytest.raises(BankError):
        bank.transfer(999_999_99, "sarah", "checking")
    assert bank.balance("checking") == 248_350


@pytest.mark.parametrize("cents", [0, -100])
def test_transfer_refuses_a_non_positive_amount(bank: SqliteBank, cents: int) -> None:
    with pytest.raises(BankError):
        bank.transfer(cents, "sarah", "checking")
    assert bank.balance("checking") == 248_350


# --- spending ----------------------------------------------------------------


def test_an_unknown_period_raises_rather_than_summing_to_zero(bank: SqliteBank) -> None:
    # Returning 0 would have the bot confidently answer "nothing" -- a wrong
    # figure delivered through the very mechanism meant to prevent wrong figures.
    with pytest.raises(BankError):
        bank.spending("october")


def test_the_unknown_period_message_names_the_valid_ones(bank: SqliteBank) -> None:
    # The executor surfaces this text to the model as a hint, so it has to
    # list what the model is allowed to ask for.
    with pytest.raises(BankError, match="this week"):
        bank.spending("october")


def test_each_valid_period_is_accepted(bank: SqliteBank) -> None:
    for period in PERIODS:
        assert isinstance(bank.spending(period), int)


def test_spending_is_reported_as_a_positive_total(bank: SqliteBank) -> None:
    # Money out is stored negative; "how much did I spend" wants a positive
    # number, the way a person means it.
    assert bank.spending("this week") > 0


def test_spending_can_be_filtered_by_merchant(bank: SqliteBank) -> None:
    everything = bank.spending("this week")
    netflix = bank.spending("this week", merchant="netflix")
    assert 0 < netflix < everything


def test_period_range_is_half_open() -> None:
    now = datetime(2026, 10, 22, 15, 30)
    start, end = period_range("today", now)
    assert start == datetime(2026, 10, 22, 0, 0)
    assert end == datetime(2026, 10, 23, 0, 0)


# --- cards and bills ---------------------------------------------------------


def test_freeze_card(bank: SqliteBank) -> None:
    assert bank.freeze_card("4417").frozen is True
    assert next(c for c in bank.cards() if c.last4 == "4417").frozen is True


def test_freezing_an_unknown_card_is_refused(bank: SqliteBank) -> None:
    with pytest.raises(BankError):
        bank.freeze_card("0000")


def test_pay_bill_deducts_and_marks_it_paid(bank: SqliteBank) -> None:
    assert bank.pay_bill("electricity") == 7_250
    assert bank.balance("checking") == 248_350 - 7_250
    assert next(b for b in bank.bills() if b.name == "electricity").paid is True


def test_pay_bill_twice_refuses_the_second_time(bank: SqliteBank) -> None:
    bank.pay_bill("electricity")
    with pytest.raises(BankError):
        bank.pay_bill("electricity")


def test_paying_an_unknown_bill_is_refused(bank: SqliteBank) -> None:
    with pytest.raises(BankError):
        bank.pay_bill("yacht mooring")


def test_order_card_for_an_unknown_account_is_refused(bank: SqliteBank) -> None:
    with pytest.raises(BankError):
        bank.order_card("offshore")


# --- audit log and reset -----------------------------------------------------


def test_log_turn_is_recorded(bank: SqliteBank) -> None:
    bank.log_turn("s1", 1, "hi", "hello", None, False, None, 12.0)
    assert bank.turn_count() == 1


def test_guard_hits_are_counted_separately(bank: SqliteBank) -> None:
    bank.log_turn("s1", 1, "hi", "hello", None, False, None, 12.0)
    bank.log_turn("s1", 2, "balance?", "safe text", None, True, "invented", 20.0)
    assert bank.turn_count() == 2
    assert bank.guard_hit_count() == 1


def test_reset_restores_the_seeded_state_but_keeps_the_audit_log(bank: SqliteBank) -> None:
    bank.transfer(5_000, "sarah", "checking")
    bank.pay_bill("electricity")
    bank.freeze_card("4417")
    bank.log_turn("s1", 1, "hi", "hello", None, False, None, 12.0)

    bank.reset()

    assert bank.balance("checking") == 248_350
    assert next(b for b in bank.bills() if b.name == "electricity").paid is False
    assert next(c for c in bank.cards() if c.last4 == "4417").frozen is False
    assert bank.turn_count() == 1


# --- money formatting --------------------------------------------------------


@pytest.mark.parametrize(
    ("cents", "expected"),
    [(1_902_000, "$19,020.00"), (248_350, "$2,483.50"), (0, "$0.00"), (5, "$0.05")],
)
def test_format_money(cents: int, expected: str) -> None:
    assert format_money(cents) == expected
