"""What a card is, what is printed on it, and the PIN in front of that.

Every test opens its own `:memory:` database, so a PIN set in one is
never visible to another.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from tests import support
from voicebot.bank.base import PIN_TRIES, BankError
from voicebot.bank.sqlite_bank import DEMO_PIN, SqliteBank


def test_freeze_card(bank: SqliteBank) -> None:
    assert bank.freeze_card("4417").frozen is True
    assert next(c for c in bank.cards() if c.last4 == "4417").frozen is True


def test_freezing_an_unknown_card_is_refused(bank: SqliteBank) -> None:
    with pytest.raises(BankError):
        bank.freeze_card("0000")


def test_order_card_for_an_unknown_account_is_refused(bank: SqliteBank) -> None:
    with pytest.raises(BankError):
        bank.order_card("offshore")


def test_a_card_knows_what_kind_it_is(bank: SqliteBank) -> None:
    kinds = {card.last4: card.kind for card in bank.cards()}
    assert kinds["4417"] == "debit"
    assert kinds["9021"] == "credit"


def test_a_card_carries_the_name_on_it(bank: SqliteBank) -> None:
    assert all(card.holder for card in bank.cards())


def test_a_new_card_allows_everything(bank: SqliteBank) -> None:
    card = next(c for c in bank.cards() if c.last4 == "4417")
    assert card.controls == {"ecommerce": True, "retail": True, "atm": True}


def test_a_control_can_be_turned_off(bank: SqliteBank) -> None:
    card = bank.set_card_control("4417", "ecommerce", False)

    assert card.controls["ecommerce"] is False
    assert card.controls["retail"] is True


def test_a_control_survives_being_read_back(bank: SqliteBank) -> None:
    bank.set_card_control("4417", "atm", False)

    stored = next(c for c in bank.cards() if c.last4 == "4417")
    assert stored.controls["atm"] is False


def test_an_unknown_control_is_refused(bank: SqliteBank) -> None:
    with pytest.raises(BankError, match="ecommerce"):
        bank.set_card_control("4417", "teleportation", False)


def test_a_control_on_an_unknown_card_is_refused(bank: SqliteBank) -> None:
    with pytest.raises(BankError):
        bank.set_card_control("0000", "atm", False)


def test_a_bank_file_made_before_cards_had_controls_gains_them(tmp_path) -> None:
    """The columns are added to a database that predates them.

    `CREATE TABLE IF NOT EXISTS` does nothing to a table that already exists,
    so without this an older bank file would keep the old four columns and
    every read of a control would fail.
    """
    import sqlite3

    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript(
        "CREATE TABLE accounts (name TEXT PRIMARY KEY, balance_cents INTEGER NOT NULL);"
        "CREATE TABLE cards (last4 TEXT PRIMARY KEY, account TEXT NOT NULL, "
        "frozen INTEGER NOT NULL DEFAULT 0);"
        "INSERT INTO accounts VALUES ('checking', 100);"
        "INSERT INTO cards VALUES ('1234', 'checking', 0);"
    )
    old.commit()
    old.close()

    bank = SqliteBank(str(path))

    card = next(c for c in bank.cards() if c.last4 == "1234")
    assert card.controls == {"ecommerce": True, "retail": True, "atm": True}


#
# Rent, groceries, the wifi and the subscriptions are the same kind of thing
# as the electricity bill: money that leaves every month whether or not
# anybody thinks about it. They live in one table so there is one answer to
# "what do I owe", and so that everything on the screen is payable by voice.


def test_a_card_has_a_number_ending_in_its_last_four(bank: SqliteBank) -> None:
    for card in bank.cards():
        assert len(card.number) == 16
        assert card.number.isdigit()
        assert card.number.endswith(card.last4)


def test_every_card_has_its_own_number(bank: SqliteBank) -> None:
    numbers = {card.number for card in bank.cards()}
    assert len(numbers) == len(bank.cards())


def test_a_card_is_valid_between_two_dates(bank: SqliteBank) -> None:
    for card in bank.cards():
        assert len(card.valid_from) == 5 and card.valid_from[2] == "/"
        assert len(card.valid_to) == 5 and card.valid_to[2] == "/"
        assert card.valid_to > card.valid_from


def test_the_bank_holds_no_security_code(bank: SqliteBank) -> None:
    """Asserted rather than assumed: it must not creep back in."""
    columns = support.columns(bank, "cards")
    assert not columns & {"cvv", "cvc", "security_code", "pin"}


def test_a_replacement_gets_a_number_of_its_own(bank: SqliteBank) -> None:
    existing = {card.number for card in bank.cards()}

    issued = bank.order_card("checking")

    assert issued.number not in existing
    assert issued.number.endswith(issued.last4)


def test_a_bank_file_made_before_numbers_gains_them(tmp_path) -> None:
    import sqlite3

    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript(
        "CREATE TABLE accounts (name TEXT PRIMARY KEY, balance_cents INTEGER NOT NULL);"
        "CREATE TABLE cards (last4 TEXT PRIMARY KEY, account TEXT NOT NULL, "
        "frozen INTEGER NOT NULL DEFAULT 0);"
        "INSERT INTO accounts VALUES ('checking', 100);"
        "INSERT INTO cards VALUES ('1234', 'checking', 0);"
    )
    old.commit()
    old.close()

    card = next(c for c in SqliteBank(str(path)).cards() if c.last4 == "1234")
    assert card.number.endswith("1234")
    assert card.valid_to


#
# A credential, so it is stored the way credentials are stored: salted, hashed
# and never readable back. Three wrong tries locks it for a while, because a
# four-digit secret with unlimited guesses is a four-digit secret that anyone
# has.


def test_the_seeded_pin_opens_a_card(bank: SqliteBank) -> None:
    assert bank.check_pin("4417", "1234") is True


def test_a_wrong_pin_does_not(bank: SqliteBank) -> None:
    assert bank.check_pin("4417", "9999") is False


def test_the_pin_is_not_stored_in_the_clear(bank: SqliteBank) -> None:
    row = support.rows(bank, "SELECT * FROM cards WHERE last4 = '4417'")[0]
    stored = " ".join(str(value) for value in row.values())

    assert "1234" not in stored
    assert row["pin_hash"] and row["pin_salt"]


def test_two_cards_with_the_same_pin_do_not_look_alike(bank: SqliteBank) -> None:
    """Salted per card, so one hash says nothing about another."""
    rows = support.rows(bank, "SELECT pin_hash, pin_salt FROM cards")
    assert len({r["pin_salt"] for r in rows}) == len(rows)
    assert len({r["pin_hash"] for r in rows}) == len(rows)


def test_three_wrong_tries_lock_the_pin(bank: SqliteBank) -> None:
    for _ in range(3):
        bank.check_pin("4417", "0000")

    assert bank.pin_locked("4417") is True
    with pytest.raises(BankError, match="locked"):
        bank.check_pin("4417", "1234")


def test_a_right_pin_forgets_the_wrong_ones(bank: SqliteBank) -> None:
    bank.check_pin("4417", "0000")
    bank.check_pin("4417", "0000")

    assert bank.check_pin("4417", "1234") is True

    bank.check_pin("4417", "0000")
    assert bank.pin_locked("4417") is False, "the count started again"


def test_a_lock_lifts_on_its_own(bank: SqliteBank) -> None:

    for _ in range(3):
        bank.check_pin("4417", "0000")

    later = datetime.now() + timedelta(minutes=10)
    assert bank.pin_locked("4417", now=later) is False
    assert bank.check_pin("4417", "1234", now=later) is True


def test_a_pin_can_be_changed_with_the_old_one(bank: SqliteBank) -> None:
    bank.set_pin("4417", "1234", "8765")

    assert bank.check_pin("4417", "8765") is True
    assert bank.check_pin("4417", "1234") is False


def test_a_pin_cannot_be_changed_without_the_old_one(bank: SqliteBank) -> None:
    with pytest.raises(BankError):
        bank.set_pin("4417", "0000", "8765")

    assert bank.check_pin("4417", "1234") is True


def test_a_new_pin_has_to_be_four_digits(bank: SqliteBank) -> None:
    for bad in ("123", "12345", "abcd", "", "12 4"):
        with pytest.raises(BankError):
            bank.set_pin("4417", "1234", bad)


def test_a_replacement_card_carries_the_pin_over(bank: SqliteBank) -> None:
    """A new card for the same person, not a new person."""
    issued = bank.order_card("checking")
    assert bank.check_pin(issued.last4, "1234") is True


def test_changing_one_cards_pin_leaves_the_other_alone(bank: SqliteBank) -> None:
    bank.set_pin("4417", "1234", "8765")
    assert bank.check_pin("9021", "1234") is True


def test_an_older_bank_file_gains_the_name_on_the_card(tmp_path) -> None:
    """The column was added with an empty default and nothing filled it, so
    the details screen showed a card with no name on it.

    Third time a migration has added a column and left the rows alone. The
    number, the dates and the PIN were all filled here already; the holder
    was simply missed.
    """
    import sqlite3

    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript(
        "CREATE TABLE accounts (name TEXT PRIMARY KEY, balance_cents INTEGER NOT NULL);"
        "CREATE TABLE cards (last4 TEXT PRIMARY KEY, account TEXT NOT NULL, "
        "frozen INTEGER NOT NULL DEFAULT 0);"
        "INSERT INTO accounts VALUES ('checking', 100);"
        "INSERT INTO cards VALUES ('1234', 'checking', 0);"
    )
    old.commit()
    old.close()

    card = next(c for c in SqliteBank(str(path)).cards() if c.last4 == "1234")
    assert card.holder == "Rehan Ali Ch"


def test_a_name_already_on_a_card_is_left_alone(tmp_path) -> None:
    """Only an empty one is filled: a name somebody set is theirs."""
    import sqlite3

    path = tmp_path / "named.db"
    old = sqlite3.connect(path)
    old.executescript(
        "CREATE TABLE accounts (name TEXT PRIMARY KEY, balance_cents INTEGER NOT NULL);"
        "CREATE TABLE cards (last4 TEXT PRIMARY KEY, account TEXT NOT NULL, "
        "frozen INTEGER NOT NULL DEFAULT 0, holder TEXT NOT NULL DEFAULT '');"
        "INSERT INTO accounts VALUES ('checking', 100);"
        "INSERT INTO cards VALUES ('1234', 'checking', 0, 'SOMEBODY ELSE');"
    )
    old.commit()
    old.close()

    card = next(c for c in SqliteBank(str(path)).cards() if c.last4 == "1234")
    assert card.holder == "SOMEBODY ELSE"


# --- a reset has to put the cards back too ------------------------------------
#
# `reset` empties the tables and seeds them again. Seeding writes the rows;
# it is `_print_cards` that gives a card its number, the name on it and a
# PIN. Leaving that out left both cards with a blank number and no PIN at
# all, so the demo PIN was refused and View details showed an empty card --
# a migration is the rows as well as the columns, and so is a reset.


def test_the_demo_pin_still_opens_a_card_after_a_reset(bank: SqliteBank) -> None:
    assert bank.check_pin("4417", DEMO_PIN) is True

    bank.reset()

    assert bank.check_pin("4417", DEMO_PIN) is True


def test_a_card_keeps_its_number_and_name_after_a_reset(bank: SqliteBank) -> None:
    bank.reset()

    card = next(c for c in bank.cards() if c.last4 == "4417")
    assert card.number.endswith("4417")
    assert card.holder
    assert card.valid_from and card.valid_to


def test_every_card_is_usable_after_a_reset(bank: SqliteBank) -> None:
    bank.reset()

    for card in bank.cards():
        assert bank.check_pin(card.last4, DEMO_PIN) is True


def test_a_reset_lifts_a_lock_earned_before_it(bank: SqliteBank) -> None:
    """Back to the opening state means the card works, not that it is barred."""
    for _ in range(PIN_TRIES):
        bank.check_pin("4417", "9999")
    assert bank.pin_locked("4417") is True

    bank.reset()

    assert bank.pin_locked("4417") is False
    assert bank.check_pin("4417", DEMO_PIN) is True
