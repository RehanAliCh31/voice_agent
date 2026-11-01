"""Balances, transfers, the ledger and a month's statement -- everything
that moves money or reports on money that moved.

These exercise real SQL against a `:memory:` database rather than a mock,
because the arithmetic is the part that must not be wrong.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from tests import support
from voicebot.bank.base import PERIODS, BankError, format_money, period_range
from voicebot.bank.sqlite_bank import SqliteBank


def test_seeded_balances(bank: SqliteBank) -> None:
    assert bank.balance("checking") == 248_350
    assert bank.balance("savings") == 1_902_000


def test_an_unknown_account_is_refused(bank: SqliteBank) -> None:
    with pytest.raises(BankError):
        bank.balance("offshore")


def test_the_seeded_cards_and_bills_are_listed(bank: SqliteBank) -> None:
    assert {card.last4 for card in bank.cards()} == {"4417", "9021"}
    assert "electricity" in {bill.name for bill in bank.bills()}


# Mid-month because on a Monday the seed's "this week" charges are today's.
@pytest.mark.mid_month
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


@pytest.mark.parametrize(
    ("cents", "expected"),
    [(1_902_000, "$19,020.00"), (248_350, "$2,483.50"), (0, "$0.00"), (5, "$0.05")],
)
def test_format_money(cents: int, expected: str) -> None:
    assert format_money(cents) == expected


#
# An administrative operation, not something the assistant can do. A bank
# assistant that can create money is not a bank assistant, so there is no tool
# for this -- only the command line.


def test_a_deposit_raises_the_balance(bank: SqliteBank) -> None:
    assert bank.deposit(50_000, "checking") == 248_350 + 50_000
    assert bank.balance("checking") == 298_350


def test_a_deposit_writes_a_ledger_row(bank: SqliteBank) -> None:
    before = bank.spending("today")
    bank.deposit(50_000, "checking", source="manual top-up")
    # Money in is a positive row, so it does not count as spending.
    assert bank.spending("today") == before


def test_a_deposit_appears_in_the_ledger_as_money_in(bank: SqliteBank) -> None:
    bank.deposit(50_000, "checking", source="manual top-up")
    row = support.rows(bank, "SELECT merchant, cents FROM transactions ORDER BY id DESC LIMIT 1")[0]
    assert row["cents"] == 50_000
    assert row["merchant"] == "manual top-up"


@pytest.mark.parametrize("cents", [0, -100])
def test_a_deposit_must_be_positive(bank: SqliteBank, cents: int) -> None:
    with pytest.raises(BankError):
        bank.deposit(cents, "checking")
    assert bank.balance("checking") == 248_350


def test_a_deposit_to_an_unknown_account_is_refused(bank: SqliteBank) -> None:
    with pytest.raises(BankError):
        bank.deposit(50_000, "offshore")


def test_a_deposit_restores_an_emptied_account(bank: SqliteBank) -> None:
    bank.transfer(248_350, "sarah", "checking")
    assert bank.balance("checking") == 0
    bank.deposit(100_000, "checking", source="manual top-up")
    assert bank.balance("checking") == 100_000


#
# `spending` totals the same table. These read it, which is what answers "what
# was that charge" rather than "how much in all".


def test_transactions_come_back_newest_first(bank: SqliteBank) -> None:
    rows = bank.transactions()
    stamps = [row.occurred_at for row in rows]
    assert stamps == sorted(stamps, reverse=True)


def test_transactions_are_limited(bank: SqliteBank) -> None:
    assert len(bank.transactions(limit=2)) == 2


def test_transactions_can_be_filtered_by_merchant(bank: SqliteBank) -> None:
    rows = bank.transactions(merchant="netflix")
    assert rows and {row.merchant for row in rows} == {"netflix"}


def test_transactions_can_be_filtered_by_account(bank: SqliteBank) -> None:
    rows = bank.transactions(account="checking")
    assert rows and {row.account for row in rows} == {"checking"}


def test_an_unknown_account_is_refused_for_transactions(bank: SqliteBank) -> None:
    with pytest.raises(BankError):
        bank.transactions(account="offshore")


def test_money_out_is_stored_negative_and_reported_as_spent(bank: SqliteBank) -> None:
    """The sign is the direction. A caller should not have to infer it."""
    netflix = bank.transactions(merchant="netflix")[0]
    assert netflix.cents < 0
    assert netflix.spent is True


def test_a_deposit_shows_in_the_ledger_as_money_in(bank: SqliteBank) -> None:
    bank.deposit(25_000, "checking", "cheque")

    newest = bank.transactions(limit=1)[0]
    assert newest.merchant == "cheque"
    assert newest.cents == 25_000
    assert newest.spent is False


#
# `transfer` only ever debits: money goes to a payee and leaves the bank. Asked
# to "send 500 to savings from checking" the model called it with payee
# "savings", so $500 left checking and arrived nowhere. Both the confirmation
# gate and the numeric guard passed it, because neither is wrong about what it
# checks -- there was simply no code that credited a destination.


def test_moving_money_debits_one_account_and_credits_the_other(bank: SqliteBank) -> None:
    before = (bank.balance("checking"), bank.balance("savings"))

    out, into = bank.move(50_000, "checking", "savings")

    assert (out, into) == (before[0] - 50_000, before[1] + 50_000)
    assert (bank.balance("checking"), bank.balance("savings")) == (out, into)


def test_moving_money_conserves_it(bank: SqliteBank) -> None:
    """The property the old code broke: nothing is created, nothing is lost."""
    before = bank.balance("checking") + bank.balance("savings")

    bank.move(50_000, "checking", "savings")

    assert bank.balance("checking") + bank.balance("savings") == before


def test_a_move_writes_both_sides_to_the_ledger(bank: SqliteBank) -> None:
    bank.move(50_000, "checking", "savings")

    rows = bank.transactions(limit=2)
    assert {(r.account, r.cents) for r in rows} == {
        ("checking", -50_000),
        ("savings", 50_000),
    }


def test_a_move_to_the_same_account_is_refused(bank: SqliteBank) -> None:
    with pytest.raises(BankError):
        bank.move(50_000, "checking", "checking")


def test_a_move_of_more_than_there_is_changes_nothing(bank: SqliteBank) -> None:
    before = (bank.balance("checking"), bank.balance("savings"))

    with pytest.raises(BankError):
        bank.move(999_999_999, "checking", "savings")

    assert (bank.balance("checking"), bank.balance("savings")) == before


def test_a_move_to_an_unknown_account_is_refused(bank: SqliteBank) -> None:
    with pytest.raises(BankError):
        bank.move(50_000, "checking", "offshore")


def test_a_move_of_nothing_is_refused(bank: SqliteBank) -> None:
    with pytest.raises(BankError):
        bank.move(0, "checking", "savings")


def test_the_bank_knows_its_payees(bank: SqliteBank) -> None:
    assert {p.name for p in bank.payees()} >= {"sarah", "ali"}


def test_a_transfer_to_an_account_name_is_refused(bank: SqliteBank) -> None:
    """The bug itself. An account is not a payee, and saying so is the fix."""
    before = (bank.balance("checking"), bank.balance("savings"))

    with pytest.raises(BankError):
        bank.transfer(50_000, "savings", "checking")

    assert (bank.balance("checking"), bank.balance("savings")) == before


def test_a_transfer_to_an_unknown_payee_changes_nothing(bank: SqliteBank) -> None:
    before = bank.balance("checking")

    with pytest.raises(BankError):
        bank.transfer(50_000, "a stranger", "checking")

    assert bank.balance("checking") == before


def test_the_refusal_says_who_can_be_paid(bank: SqliteBank) -> None:
    with pytest.raises(BankError, match="sarah"):
        bank.transfer(50_000, "a stranger", "checking")


def test_a_transfer_to_a_known_payee_still_works(bank: SqliteBank) -> None:
    """The existing behaviour, which must survive the fix."""
    remaining = bank.transfer(5_000, "sarah", "checking")
    assert remaining == 243_350


#
# The bot cannot do everything, and the cases it must not try are the ones
# where being wrong costs most: fraud, complaints, money trouble. A handoff
# records that the conversation left the bot, and points at the turns already
# logged for the session rather than copying them, so the person picking it up
# reads what was actually said.


# Mid-month because on the first days of a month everything the seed spent
# can be last month's, and this month's statement is empty.
@pytest.mark.mid_month
def test_a_statement_holds_the_months_movements(bank: SqliteBank) -> None:
    now = support.MID_MONTH
    statement = bank.statement(now.year, now.month)

    assert statement.rows
    assert all(row.occurred_at.startswith(now.strftime("%Y-%m")) for row in statement.rows)


@pytest.mark.mid_month
def test_a_statement_is_newest_first(bank: SqliteBank) -> None:
    now = support.MID_MONTH
    stamps = [row.occurred_at for row in bank.statement(now.year, now.month).rows]

    assert stamps == sorted(stamps, reverse=True)


@pytest.mark.mid_month
def test_a_statement_totals_both_directions(bank: SqliteBank) -> None:
    now = support.MID_MONTH
    bank.deposit(25_000, "checking", "manual top-up")

    statement = bank.statement(now.year, now.month)

    assert statement.incoming_cents == 25_000
    assert statement.outgoing_cents > 0
    assert all(total >= 0 for total in (statement.incoming_cents, statement.outgoing_cents))


def test_a_month_with_nothing_in_it_is_empty_not_an_error(bank: SqliteBank) -> None:
    statement = bank.statement(1999, 1)

    assert statement.rows == ()
    assert statement.incoming_cents == 0
    assert statement.outgoing_cents == 0


def test_an_impossible_month_is_refused(bank: SqliteBank) -> None:
    with pytest.raises(BankError):
        bank.statement(2026, 13)


#
# A card is more than four digits and a frozen flag once there is a screen
# drawing it. The three controls are real columns rather than decoration: a
# switch on screen that nothing can change is a lie told in CSS.


# --- a period for the rows, not just the totals -------------------------------
#
# The Transactions screen shows one calendar month. The bot reading "the
# newest five" instead listed two charges from last month alongside three
# from this one and called them all recent, so the screen and the sentence
# disagreed about the same account.


# Mid-month because early in a month "this week" and "yesterday" can be part
# of last month, and then last month holds every charge the seed made.
@pytest.mark.mid_month
def test_transactions_can_be_limited_to_a_period(bank: SqliteBank) -> None:
    last_month = {t.merchant for t in bank.transactions(limit=20, period="last month")}
    this_week = {t.merchant for t in bank.transactions(limit=20, period="this week")}

    assert last_month == {"amazon", "tesco"}
    # Seeded into this week by name. Yesterday's charge is deliberately not
    # asserted either way: whether it falls in this week depends on the day
    # of the week, which is not what this test is about.
    assert {"netflix", "uber"} <= this_week
    assert not last_month & this_week


# Mid-month because on a Monday the seed's "this week" charges are today's.
@pytest.mark.mid_month
def test_a_period_with_nothing_in_it_is_empty_rather_than_wrong(bank: SqliteBank) -> None:
    # Nothing was charged today in the seed. An empty list is the honest
    # answer; falling back to "the newest five" would put last month's
    # charges under today's heading.
    assert bank.transactions(limit=20, period="today") == ()


def test_an_unknown_period_for_rows_names_the_ones_that_work(bank: SqliteBank) -> None:
    with pytest.raises(BankError, match="this week"):
        bank.transactions(period="october")


def test_every_period_is_accepted_for_rows(bank: SqliteBank) -> None:
    for period in PERIODS:
        assert isinstance(bank.transactions(period=period), tuple)


def test_without_a_period_the_ledger_behaves_as_it_did(bank: SqliteBank) -> None:
    """The bank keeps its old answer; it is the tool above that now insists."""
    assert len(bank.transactions(limit=20)) == 5
