"""The bank, backed by SQLite.

One connection per instance, shared by every session in the process: this is
one customer's bank, and a phone and a laptop looking at the same account is
the realistic case rather than a bug.

Every balance change and its ledger row happen inside one transaction, so an
interruption leaves neither.
"""

from __future__ import annotations

import secrets
import sqlite3
from datetime import datetime
from pathlib import Path

from voicebot.bank.base import (
    PERIODS,
    Account,
    Bank,
    BankError,
    Bill,
    Card,
    Payee,
    Transaction,
    account_not_payee_message,
    format_money,
    period_range,
    unknown_payee_message,
    unknown_period_message,
)
from voicebot.bank.seed import clear, seed, seed_payees
from voicebot.utils.logging import get_logger

log = get_logger("bank.sqlite")

SCHEMA = Path(__file__).parent / "schema.sql"


class SqliteBank(Bank):
    """A bank in a SQLite database. `:memory:` for tests, a file for serving."""

    def __init__(self, path: str = ":memory:", seed_if_empty: bool = True) -> None:
        self.path = path
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False because the API serves requests on a thread
        # pool; every write below is wrapped in a transaction, and SQLite
        # serialises them.
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(SCHEMA.read_text(encoding="utf-8"))
        if seed_if_empty and not self._db.execute("SELECT 1 FROM accounts").fetchone():
            seed(self._db)
            log.info("seeded a new bank at %s", path)
        elif seed_if_empty:
            # The schema runs on every open, so an older bank file now has an
            # empty payees table. Left empty, every transfer would be refused
            # for want of a payee that was never inserted.
            seed_payees(self._db)

    # --- reads ---------------------------------------------------------------

    def accounts(self) -> tuple[Account, ...]:
        rows = self._db.execute("SELECT name, balance_cents FROM accounts ORDER BY name")
        return tuple(Account(name=r["name"], balance_cents=r["balance_cents"]) for r in rows)

    def balance(self, account: str) -> int:
        row = self._db.execute(
            "SELECT balance_cents FROM accounts WHERE name = ?", (self._account(account),)
        ).fetchone()
        assert row is not None  # _account already proved it exists
        return int(row["balance_cents"])

    def cards(self) -> tuple[Card, ...]:
        rows = self._db.execute("SELECT last4, account, frozen FROM cards ORDER BY last4")
        return tuple(
            Card(last4=r["last4"], account=r["account"], frozen=bool(r["frozen"])) for r in rows
        )

    def bills(self) -> tuple[Bill, ...]:
        rows = self._db.execute("SELECT name, amount_cents, paid FROM bills ORDER BY name")
        return tuple(
            Bill(name=r["name"], amount_cents=r["amount_cents"], paid=bool(r["paid"])) for r in rows
        )

    def payees(self) -> tuple[Payee, ...]:
        rows = self._db.execute("SELECT name FROM payees ORDER BY name")
        return tuple(Payee(name=r["name"]) for r in rows)

    def transactions(
        self,
        limit: int = 5,
        account: str | None = None,
        merchant: str | None = None,
    ) -> tuple[Transaction, ...]:
        sql = "SELECT account, merchant, cents, occurred_at FROM transactions"
        clauses: list[str] = []
        params: list[object] = []

        if account:
            clauses.append("account = ?")
            params.append(self._account(account))
        if merchant:
            clauses.append("merchant = ?")
            params.append(merchant.lower())
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)

        # id breaks ties: the seed writes several rows inside one second, and
        # a timestamp alone would order them arbitrarily between runs.
        sql += " ORDER BY occurred_at DESC, id DESC LIMIT ?"
        params.append(max(1, limit))

        return tuple(
            Transaction(
                account=row["account"],
                merchant=row["merchant"],
                cents=int(row["cents"]),
                occurred_at=row["occurred_at"],
            )
            for row in self._db.execute(sql, params).fetchall()
        )

    def spending(self, period: str, merchant: str | None = None) -> int:
        if period not in PERIODS:
            raise BankError(unknown_period_message(period))
        start, end = period_range(period, datetime.now())

        sql = (
            "SELECT COALESCE(SUM(-cents), 0) AS total FROM transactions "
            "WHERE cents < 0 AND occurred_at >= ? AND occurred_at < ?"
        )
        params: list[object] = [_stamp(start), _stamp(end)]
        if merchant:
            sql += " AND merchant = ?"
            params.append(merchant.lower())

        row = self._db.execute(sql, params).fetchone()
        return int(row["total"])

    # --- writes --------------------------------------------------------------

    def transfer(self, cents: int, payee: str, from_account: str) -> int:
        if cents <= 0:
            raise BankError("The amount has to be more than zero.")
        account = self._account(from_account)
        payee = self._payee(payee)

        # The sufficiency test is the UPDATE's own WHERE clause rather than a
        # separate read. Checking first and writing second lets two concurrent
        # transfers both pass the check; the loser then violates the CHECK
        # constraint and raises instead of being refused in words.
        with self._db:
            changed = self._db.execute(
                "UPDATE accounts SET balance_cents = balance_cents - ? "
                "WHERE name = ? AND balance_cents >= ?",
                (cents, account, cents),
            ).rowcount
            if not changed:
                available = self.balance(account)
                raise BankError(
                    f"There isn't enough in {account}. "
                    f"The balance is {format_money(available)}."
                )
            self._ledger(account, payee.lower(), -cents)
        return self.balance(account)

    def move(self, cents: int, from_account: str, to_account: str) -> tuple[int, int]:
        if cents <= 0:
            raise BankError("The amount has to be more than zero.")
        source = self._account(from_account)
        target = self._account(to_account)
        if source == target:
            raise BankError(f"{source} and {target} are the same account.")

        # Both sides inside one transaction. The raise below rolls the debit
        # back, so there is no state where the money has left and not arrived
        # -- which is the whole failure this replaces.
        with self._db:
            changed = self._db.execute(
                "UPDATE accounts SET balance_cents = balance_cents - ? "
                "WHERE name = ? AND balance_cents >= ?",
                (cents, source, cents),
            ).rowcount
            if not changed:
                available = self.balance(source)
                raise BankError(
                    f"There isn't enough in {source}. " f"The balance is {format_money(available)}."
                )
            self._db.execute(
                "UPDATE accounts SET balance_cents = balance_cents + ? WHERE name = ?",
                (cents, target),
            )
            # Each side records the other, so the history reads from either
            # account and says where the money went or came from.
            self._ledger(source, target, -cents)
            self._ledger(target, source, cents)

        return self.balance(source), self.balance(target)

    def pay_bill(self, bill: str) -> int:
        row = self._db.execute(
            "SELECT amount_cents, paid FROM bills WHERE name = ?", (bill.lower(),)
        ).fetchone()
        if row is None:
            known = ", ".join(b.name for b in self.bills())
            raise BankError(f"I don't have a {bill} bill on file. I have: {known}.")
        if row["paid"]:
            raise BankError(f"Your {bill} bill is already paid.")

        amount = int(row["amount_cents"])

        with self._db:
            changed = self._db.execute(
                "UPDATE accounts SET balance_cents = balance_cents - ? "
                "WHERE name = 'checking' AND balance_cents >= ?",
                (amount, amount),
            ).rowcount
            if not changed:
                available = self.balance("checking")
                raise BankError(
                    f"There isn't enough in checking to cover that. The bill is "
                    f"{format_money(amount)} and the balance is {format_money(available)}."
                )
            self._db.execute("UPDATE bills SET paid = 1 WHERE name = ?", (bill.lower(),))
            self._ledger("checking", bill.lower(), -amount)
        return amount

    def freeze_card(self, last4: str) -> Card:
        with self._db:
            changed = self._db.execute(
                "UPDATE cards SET frozen = 1 WHERE last4 = ?", (last4,)
            ).rowcount
        if not changed:
            known = ", ".join(c.last4 for c in self.cards())
            raise BankError(f"I can't find a card ending {last4}. I have: {known}.")
        return next(card for card in self.cards() if card.last4 == last4)

    def order_card(self, account: str) -> Card:
        """Issue a new card and return it.

        Returns the card rather than nothing so the bot can tell the customer
        which one is coming; a tool that changes nothing cannot be meaningfully
        confirmed.
        """
        name = self._account(account)
        existing = {card.last4 for card in self.cards()}
        last4 = f"{secrets.randbelow(10_000):04d}"
        while last4 in existing:
            last4 = f"{secrets.randbelow(10_000):04d}"
        with self._db:
            self._db.execute("INSERT INTO cards VALUES (?, ?, 0)", (last4, name))
        return Card(last4=last4, account=name, frozen=False)

    def deposit(self, cents: int, account: str, source: str = "deposit") -> int:
        """Pay money in. Writes a positive ledger row, so the history stays true."""
        if cents <= 0:
            raise BankError("A deposit has to be more than zero.")
        name = self._account(account)

        with self._db:
            self._db.execute(
                "UPDATE accounts SET balance_cents = balance_cents + ? WHERE name = ?",
                (cents, name),
            )
            self._ledger(name, source.lower(), cents)
        return self.balance(name)

    # --- audit ---------------------------------------------------------------

    def log_turn(
        self,
        session_id: str,
        turn_index: int,
        user_text: str,
        bot_text: str,
        tool_calls: str | None,
        guard_hit: bool,
        suppressed: str | None,
        latency_ms: float,
    ) -> None:
        with self._db:
            self._db.execute(
                "INSERT INTO turns (session_id, turn_index, user_text, bot_text, "
                "tool_calls, guard_hit, suppressed, latency_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    turn_index,
                    user_text,
                    bot_text,
                    tool_calls,
                    int(guard_hit),
                    suppressed,
                    latency_ms,
                ),
            )

    def turn_count(self) -> int:
        return int(self._db.execute("SELECT COUNT(*) AS n FROM turns").fetchone()["n"])

    def guard_hit_count(self) -> int:
        return int(
            self._db.execute("SELECT COUNT(*) AS n FROM turns WHERE guard_hit = 1").fetchone()["n"]
        )

    def reset(self) -> None:
        """Back to the opening state. `turns` survives -- it is the record."""
        clear(self._db)
        seed(self._db)
        log.info("bank reset to seeded state")

    # --- internals -----------------------------------------------------------

    def _account(self, name: str) -> str:
        """Validate an account name, or say what exists."""
        key = name.strip().lower()
        if self._db.execute("SELECT 1 FROM accounts WHERE name = ?", (key,)).fetchone():
            return key
        known = ", ".join(a.name for a in self.accounts())
        raise BankError(f"I don't recognise the account {name!r}. I have: {known}.")

    def _payee(self, name: str) -> str:
        """Validate a payee, or say plainly why this one will not do.

        An account name gets its own message rather than the generic one. It
        is the mistake that actually happened, and the sentence is what tells
        the model to move the money instead of sending it.
        """
        key = name.strip().lower()
        if self._db.execute("SELECT 1 FROM payees WHERE name = ?", (key,)).fetchone():
            return key
        if self._db.execute("SELECT 1 FROM accounts WHERE name = ?", (key,)).fetchone():
            raise BankError(account_not_payee_message(key))
        raise BankError(unknown_payee_message(name, tuple(p.name for p in self.payees())))

    def _ledger(self, account: str, merchant: str, cents: int) -> None:
        """Record a movement. Called inside an open transaction."""
        self._db.execute(
            "INSERT INTO transactions (account, merchant, cents, occurred_at) VALUES (?, ?, ?, ?)",
            (account, merchant, cents, _stamp(datetime.now())),
        )


def _stamp(moment: datetime) -> str:
    """One timestamp format, so string comparison in SQL orders correctly."""
    return moment.isoformat(sep=" ", timespec="seconds")
