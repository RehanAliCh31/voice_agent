"""The SQLite bank: balances, the ledger, and the audit log.

Every test opens its own `:memory:` database, so they are isolated from each
other and from any file on disk, and they exercise real SQL rather than a mock.
"""

from __future__ import annotations

from datetime import datetime, timedelta

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


# --- paying money in ---------------------------------------------------------
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
    row = bank._db.execute(
        "SELECT merchant, cents FROM transactions ORDER BY id DESC LIMIT 1"
    ).fetchone()
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


# --- the ledger, row by row --------------------------------------------------
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


# --- moving money between the customer's own accounts ------------------------
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


# --- who money may be sent to ------------------------------------------------


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


# --- handing a customer to a person ------------------------------------------
#
# The bot cannot do everything, and the cases it must not try are the ones
# where being wrong costs most: fraud, complaints, money trouble. A handoff
# records that the conversation left the bot, and points at the turns already
# logged for the session rather than copying them, so the person picking it up
# reads what was actually said.


def test_a_handoff_gets_a_reference(bank: SqliteBank) -> None:
    case = bank.open_handoff("s1", 3, "fraud", "card used in another country")

    assert case.reference
    assert case.reason == "fraud"
    assert case.session_id == "s1"
    assert case.turn_index == 3


def test_a_reference_can_be_read_aloud(bank: SqliteBank) -> None:
    """It is spoken down a phone, so it holds no letters that sound alike."""
    reference = bank.open_handoff("s1", 1, "complaint").reference

    assert reference.isupper() or reference.isdigit() or "-" in reference
    assert not set(reference) & set("OIS015")


def test_every_handoff_gets_its_own_reference(bank: SqliteBank) -> None:
    made = {bank.open_handoff("s1", i, "complaint").reference for i in range(20)}
    assert len(made) == 20


def test_a_handoff_can_be_found_by_session(bank: SqliteBank) -> None:
    bank.open_handoff("s1", 1, "fraud")
    bank.open_handoff("s2", 1, "complaint")

    assert [c.reason for c in bank.handoffs(session_id="s1")] == ["fraud"]


def test_an_unknown_reason_is_refused(bank: SqliteBank) -> None:
    with pytest.raises(BankError):
        bank.open_handoff("s1", 1, "because I felt like it")


def test_the_refusal_names_the_reasons_that_exist(bank: SqliteBank) -> None:
    with pytest.raises(BankError, match="fraud"):
        bank.open_handoff("s1", 1, "nonsense")


def test_a_handoff_survives_the_demo_reset(bank: SqliteBank) -> None:
    """A fraud report is not undone by someone starting a new conversation."""
    bank.open_handoff("s1", 1, "fraud")

    bank.reset()

    assert len(bank.handoffs()) == 1


def test_a_transaction_carries_its_id(bank: SqliteBank) -> None:
    """Without it nothing downstream can name one row rather than another."""
    rows = bank.transactions(limit=3)

    assert all(row.id for row in rows)
    assert len({row.id for row in rows}) == 3


# --- disputing a charge ------------------------------------------------------
#
# The bank files a dispute. It never settles one. Deciding whether a charge
# was really unauthorised needs evidence the bot does not have and authority
# it should not be given, so every case it opens is open, and a person closes
# it. That is also why nothing is credited back here.


def _netflix(bank: SqliteBank) -> int:
    return bank.transactions(merchant="netflix")[0].id


def test_a_dispute_is_filed_against_one_transaction(bank: SqliteBank) -> None:
    case = bank.open_dispute(_netflix(bank), "not recognised", "I cancelled this")

    assert case.reference
    assert case.transaction_id == _netflix(bank)
    assert case.reason == "not recognised"
    assert case.amount_cents == 1_099


def test_a_filed_dispute_is_always_open(bank: SqliteBank) -> None:
    """The bot has no authority to decide one, so it records none."""
    assert bank.open_dispute(_netflix(bank), "not recognised").status == "open"


def test_filing_a_dispute_moves_no_money(bank: SqliteBank) -> None:
    before = bank.balance("checking")
    bank.open_dispute(_netflix(bank), "charged twice")

    assert bank.balance("checking") == before


def test_the_same_charge_cannot_be_disputed_twice(bank: SqliteBank) -> None:
    bank.open_dispute(_netflix(bank), "not recognised")

    with pytest.raises(BankError, match="already"):
        bank.open_dispute(_netflix(bank), "charged twice")


def test_a_charge_that_does_not_exist_is_refused(bank: SqliteBank) -> None:
    with pytest.raises(BankError):
        bank.open_dispute(999_999, "not recognised")


def test_an_unknown_reason_is_refused_for_a_dispute(bank: SqliteBank) -> None:
    with pytest.raises(BankError, match="not recognised"):
        bank.open_dispute(_netflix(bank), "I just don't like it")


def test_money_arriving_cannot_be_disputed(bank: SqliteBank) -> None:
    """A dispute is about a charge. Being paid is not one."""
    bank.deposit(25_000, "checking", "cheque")
    paid_in = bank.transactions(limit=1)[0]

    with pytest.raises(BankError):
        bank.open_dispute(paid_in.id, "not recognised")


def test_a_charge_too_old_to_dispute_is_refused(bank: SqliteBank) -> None:
    """Card schemes give a window, and a bot that ignores it raises cases
    that are rejected later -- after the customer was told it was filed."""
    long_after = datetime.now() + timedelta(days=400)

    with pytest.raises(BankError, match="120 days"):
        bank.open_dispute(_netflix(bank), "not recognised", now=long_after)


def test_disputes_can_be_listed(bank: SqliteBank) -> None:
    bank.open_dispute(_netflix(bank), "not recognised")
    assert len(bank.disputes()) == 1


def test_a_dispute_survives_the_demo_reset(bank: SqliteBank) -> None:
    bank.open_dispute(_netflix(bank), "not recognised")
    bank.reset()

    assert len(bank.disputes()) == 1
