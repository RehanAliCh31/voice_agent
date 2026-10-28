"""The bills and rent that leave every month: what is owed, when it falls
due, and what paying one does.
"""

from __future__ import annotations

import pytest

from voicebot.bank.base import BankError
from voicebot.bank.sqlite_bank import SqliteBank


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


def test_every_commitment_has_a_category(bank: SqliteBank) -> None:
    from voicebot.bank.base import PLAN_CATEGORIES

    assert all(bill.category in PLAN_CATEGORIES for bill in bank.bills())


def test_the_categories_the_customer_asked_for_are_all_there(bank: SqliteBank) -> None:
    assert {bill.category for bill in bank.bills()} == {
        "bills",
        "rent",
        "groceries",
        "wifi",
        "subscriptions",
    }


def test_every_commitment_falls_due_on_a_day_of_the_month(bank: SqliteBank) -> None:
    assert all(1 <= bill.due_day <= 28 for bill in bank.bills())


def test_rent_is_the_largest_commitment(bank: SqliteBank) -> None:
    """Seed data that reads like a life rather than a list of test values."""
    largest = max(bank.bills(), key=lambda bill: bill.amount_cents)
    assert largest.category == "rent"


def test_paying_one_leaves_the_others_alone(bank: SqliteBank) -> None:
    bank.pay_bill("electricity")

    paid = {bill.name for bill in bank.bills() if bill.paid}
    assert paid == {"electricity"}


def test_a_bank_file_made_before_categories_gains_them(tmp_path) -> None:
    import sqlite3

    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript(
        "CREATE TABLE accounts (name TEXT PRIMARY KEY, balance_cents INTEGER NOT NULL);"
        "CREATE TABLE bills (name TEXT PRIMARY KEY, amount_cents INTEGER NOT NULL, "
        "paid INTEGER NOT NULL DEFAULT 0);"
        "INSERT INTO accounts VALUES ('checking', 100);"
        "INSERT INTO bills VALUES ('electricity', 7250, 0);"
    )
    old.commit()
    old.close()

    bill = next(b for b in SqliteBank(str(path)).bills() if b.name == "electricity")
    assert bill.category == "bills"
    assert bill.due_day >= 1


def test_an_older_bank_file_gains_the_commitments_it_never_had(tmp_path) -> None:
    """Adding the columns was not enough; the rows had to be put right too.

    A file written before categories existed holds the five original bills
    and none of rent, groceries or the subscriptions. The new columns took
    their defaults, so everything landed under Bills and fell due on the
    1st -- which is what the monthly screen showed.
    """
    import sqlite3

    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript(
        "CREATE TABLE accounts (name TEXT PRIMARY KEY, balance_cents INTEGER NOT NULL);"
        "CREATE TABLE bills (name TEXT PRIMARY KEY, amount_cents INTEGER NOT NULL, "
        "paid INTEGER NOT NULL DEFAULT 0);"
        "INSERT INTO accounts VALUES ('checking', 100);"
        "INSERT INTO bills VALUES ('electricity', 7250, 0), ('water', 3100, 0), "
        "('phone', 2499, 0), ('internet', 4000, 0), ('credit card', 21500, 1);"
    )
    old.commit()
    old.close()

    bills = {bill.name: bill for bill in SqliteBank(str(path)).bills()}

    assert {"rent", "groceries", "spotify", "netflix"} <= set(bills)
    assert bills["internet"].category == "wifi"
    assert bills["electricity"].due_day == 5
    assert len({bill.due_day for bill in bills.values()}) > 1


def test_putting_an_older_file_right_leaves_the_customers_own_state_alone(tmp_path) -> None:
    """Category and due day are the bank's. The amount and whether it is
    paid are the customer's, and reconciling must not touch them."""
    import sqlite3

    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript(
        "CREATE TABLE accounts (name TEXT PRIMARY KEY, balance_cents INTEGER NOT NULL);"
        "CREATE TABLE bills (name TEXT PRIMARY KEY, amount_cents INTEGER NOT NULL, "
        "paid INTEGER NOT NULL DEFAULT 0);"
        "INSERT INTO accounts VALUES ('checking', 100);"
        "INSERT INTO bills VALUES ('electricity', 9999, 1);"
    )
    old.commit()
    old.close()

    electricity = next(b for b in SqliteBank(str(path)).bills() if b.name == "electricity")

    assert electricity.amount_cents == 9999, "their bill, not the seed's"
    assert electricity.paid is True, "already paid, and it stays paid"
    assert electricity.category == "bills"
    assert electricity.due_day == 5


def test_a_new_card_can_actually_be_issued(bank: SqliteBank) -> None:
    """Broken silently by adding columns: the insert named none of them.

    Nothing caught it, because the gate tests only ever made the first call
    of a confirming tool and never let one through.
    """
    before = {card.last4 for card in bank.cards()}

    issued = bank.order_card("checking")

    assert issued.last4 not in before
    assert {card.last4 for card in bank.cards()} == before | {issued.last4}


def test_a_new_card_arrives_looking_like_a_card(bank: SqliteBank) -> None:
    issued = bank.order_card("checking")

    assert issued.holder, "it has the customer's name on it like the others"
    assert issued.kind
    assert issued.frozen is False
    assert issued.controls == {"ecommerce": True, "retail": True, "atm": True}


#
# The number and the dates are held because the details screen shows them, the
# way a bank's own app does once you are past the lock screen. They are demo
# cards and the numbers are invented.
#
# The security code is not here and is not anywhere: it is minted for one
# viewing and forgotten. A stored code is the one thing that stays wrong even
# with made-up data.
