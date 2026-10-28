"""What the bank records and what it escalates: the audit log, handing a
customer to a person, and disputing a charge.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from voicebot.bank.base import BankError
from voicebot.bank.sqlite_bank import SqliteBank


@pytest.fixture
def bank() -> SqliteBank:
    return SqliteBank()


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


#
# The transactions screen shows one calendar month: every movement in it, and
# what came in against what went out. Totalled here rather than in the page,
# so the figures on screen and the figures the bot speaks come from the same
# arithmetic.
