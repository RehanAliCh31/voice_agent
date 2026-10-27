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
    CARD_CONTROLS,
    DISPUTE_REASONS,
    DISPUTE_WINDOW_DAYS,
    HANDOFF_REASONS,
    PERIODS,
    Account,
    Bank,
    BankError,
    Bill,
    Card,
    Dispute,
    Handoff,
    Payee,
    Statement,
    Transaction,
    account_not_payee_message,
    format_money,
    period_range,
    too_old_to_dispute_message,
    unknown_control_message,
    unknown_dispute_reason_message,
    unknown_payee_message,
    unknown_period_message,
    unknown_reason_message,
)
from voicebot.bank.seed import clear, seed, seed_payees, seed_plans
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
        _widen(self._db)
        if seed_if_empty and not self._db.execute("SELECT 1 FROM accounts").fetchone():
            seed(self._db)
            _print_cards(self._db)
            log.info("seeded a new bank at %s", path)
        elif seed_if_empty:
            # The schema runs on every open, so an older bank file now has an
            # empty payees table. Left empty, every transfer would be refused
            # for want of a payee that was never inserted.
            seed_payees(self._db)
            _print_cards(self._db)
            # Likewise the commitments: the columns were added above and took
            # their defaults, which filed every bill under Bills on the 1st,
            # and the ones that never existed in the old file are still
            # missing. Columns alone were never the whole migration.
            seed_plans(self._db)

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
        rows = self._db.execute("SELECT * FROM cards ORDER BY last4")
        return tuple(_card(row) for row in rows)

    def bills(self) -> tuple[Bill, ...]:
        rows = self._db.execute("SELECT * FROM bills ORDER BY due_day, name")
        return tuple(
            Bill(
                name=row["name"],
                amount_cents=int(row["amount_cents"]),
                paid=bool(row["paid"]),
                category=row["category"],
                due_day=int(row["due_day"]),
            )
            for row in rows
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
        sql = "SELECT id, account, merchant, cents, occurred_at FROM transactions"
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

        return tuple(_transaction(row) for row in self._db.execute(sql, params).fetchall())

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

    def set_card_control(self, last4: str, control: str, enabled: bool) -> Card:
        name = control.strip().lower()
        if name not in CARD_CONTROLS:
            raise BankError(unknown_control_message(control))
        card = self._card_row(last4)

        with self._db:
            # The column name comes from CARD_CONTROLS and nowhere else, so
            # the only strings reaching this statement are the three above.
            self._db.execute(
                f"UPDATE cards SET {name} = ? WHERE last4 = ?",
                (1 if enabled else 0, card["last4"]),
            )
        return _card(self._card_row(last4))

    def _card_row(self, last4: str) -> sqlite3.Row:
        """One card, or a sentence naming the ones that exist."""
        key = last4.strip()
        row = self._db.execute("SELECT * FROM cards WHERE last4 = ?", (key,)).fetchone()
        if row is None:
            known = ", ".join(card.last4 for card in self.cards())
            raise BankError(f"I don't have a card ending {last4}. I have: {known}.")
        return row

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
        # Columns are named. The unqualified form broke the moment the table
        # grew, and nothing noticed, because the gate tests stop at the
        # read-back and never let a card order through.
        holder = next((card.holder for card in self.cards() if card.holder), "")
        valid_from, valid_to = _validity(datetime.now())
        with self._db:
            self._db.execute(
                "INSERT INTO cards (last4, account, frozen, kind, holder, "
                "number, valid_from, valid_to) VALUES (?, ?, 0, 'debit', ?, ?, ?, ?)",
                (last4, name, holder, _number_for(last4), valid_from, valid_to),
            )
        return _card(self._card_row(last4))

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

    def statement(self, year: int, month: int) -> Statement:
        if not 1 <= month <= 12:
            raise BankError(f"{month} is not a month.")

        # String comparison on a "YYYY-MM-DD HH:MM:SS" stamp: the format is
        # fixed by _stamp, so a prefix match is the whole month and needs no
        # date arithmetic or timezone opinion.
        prefix = f"{year:04d}-{month:02d}-%"
        rows = tuple(
            _transaction(row)
            for row in self._db.execute(
                "SELECT id, account, merchant, cents, occurred_at FROM transactions "
                "WHERE occurred_at LIKE ? ORDER BY occurred_at DESC, id DESC",
                (prefix,),
            ).fetchall()
        )
        return Statement(
            year=year,
            month=month,
            rows=rows,
            incoming_cents=sum(row.cents for row in rows if row.cents > 0),
            outgoing_cents=sum(-row.cents for row in rows if row.cents < 0),
        )

    def transaction(self, transaction_id: int) -> Transaction | None:
        row = self._db.execute(
            "SELECT id, account, merchant, cents, occurred_at FROM transactions WHERE id = ?",
            (transaction_id,),
        ).fetchone()
        return None if row is None else _transaction(row)

    # --- disputing a charge --------------------------------------------------

    def open_dispute(
        self,
        transaction_id: int,
        reason: str,
        note: str | None = None,
        now: datetime | None = None,
    ) -> Dispute:
        key = reason.strip().lower()
        if key not in DISPUTE_REASONS:
            raise BankError(unknown_dispute_reason_message(reason))

        row = self._db.execute(
            "SELECT id, merchant, cents, occurred_at FROM transactions WHERE id = ?",
            (transaction_id,),
        ).fetchone()
        if row is None:
            raise BankError("I can't find that charge on your account.")
        if row["cents"] >= 0:
            raise BankError(
                f"That is money paid in, not a charge. "
                f"There is nothing to dispute about {row['merchant']}."
            )

        age = (now or datetime.now()) - datetime.fromisoformat(row["occurred_at"])
        if age.days > DISPUTE_WINDOW_DAYS:
            raise BankError(too_old_to_dispute_message(age.days))

        if self._db.execute(
            "SELECT 1 FROM disputes WHERE transaction_id = ?", (transaction_id,)
        ).fetchone():
            raise BankError(
                f"The {row['merchant']} charge for "
                f"{format_money(abs(int(row['cents'])))} has already been disputed."
            )

        reference = _reference()
        with self._db:
            self._db.execute(
                "INSERT INTO disputes (reference, transaction_id, merchant, "
                "amount_cents, occurred_at, reason, note) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    reference,
                    transaction_id,
                    row["merchant"],
                    abs(int(row["cents"])),
                    row["occurred_at"],
                    key,
                    note,
                ),
            )
        return self._dispute(reference)

    def disputes(self) -> tuple[Dispute, ...]:
        rows = self._db.execute(
            "SELECT reference FROM disputes ORDER BY created_at DESC, rowid DESC"
        ).fetchall()
        return tuple(self._dispute(row["reference"]) for row in rows)

    def _dispute(self, reference: str) -> Dispute:
        """Read one case back. No join: the charge is held on the case."""
        row = self._db.execute(
            "SELECT * FROM disputes WHERE reference = ?", (reference,)
        ).fetchone()
        return Dispute(
            reference=row["reference"],
            transaction_id=int(row["transaction_id"]),
            merchant=row["merchant"],
            amount_cents=int(row["amount_cents"]),
            occurred_at=row["occurred_at"],
            reason=row["reason"],
            note=row["note"],
            status=row["status"],
            created_at=row["created_at"],
        )

    # --- handing a customer to a person --------------------------------------

    def open_handoff(
        self,
        session_id: str,
        turn_index: int,
        reason: str,
        note: str | None = None,
    ) -> Handoff:
        key = reason.strip().lower()
        if key not in HANDOFF_REASONS:
            raise BankError(unknown_reason_message(reason))

        reference = _reference()
        with self._db:
            self._db.execute(
                "INSERT INTO handoffs (reference, session_id, turn_index, reason, note) "
                "VALUES (?, ?, ?, ?, ?)",
                (reference, session_id, turn_index, key, note),
            )
        row = self._db.execute(
            "SELECT * FROM handoffs WHERE reference = ?", (reference,)
        ).fetchone()
        return _handoff(row)

    def handoffs(self, session_id: str | None = None) -> tuple[Handoff, ...]:
        sql = "SELECT * FROM handoffs"
        params: list[object] = []
        if session_id:
            sql += " WHERE session_id = ?"
            params.append(session_id)
        sql += " ORDER BY created_at DESC, rowid DESC"
        return tuple(_handoff(row) for row in self._db.execute(sql, params).fetchall())

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


# Read down a phone, so the alphabet holds nothing that sounds or looks like
# anything else in it: no O or 0, no I or 1, no S or 5.
_REFERENCE_ALPHABET = "ACDEFHJKLMNPRTUVWXY2346789"


def _reference() -> str:
    """A short case reference a customer can repeat without spelling it out."""
    body = "".join(secrets.choice(_REFERENCE_ALPHABET) for _ in range(6))
    return f"VB-{body}"


# Columns added to tables that already existed. `CREATE TABLE IF NOT EXISTS`
# does nothing to a table that is already there, so a bank file written
# before a column existed would keep the old shape and every read of the new
# one would fail. Adding them is idempotent and cheap, and it runs on every
# open rather than behind a version number nobody would remember to raise.
_ADDED: dict[str, dict[str, str]] = {
    "bills": {
        "category": "TEXT NOT NULL DEFAULT 'bills'",
        "due_day": "INTEGER NOT NULL DEFAULT 1",
    },
    "cards": {
        "number": "TEXT NOT NULL DEFAULT ''",
        "valid_from": "TEXT NOT NULL DEFAULT ''",
        "valid_to": "TEXT NOT NULL DEFAULT ''",
        "kind": "TEXT NOT NULL DEFAULT 'debit'",
        "holder": "TEXT NOT NULL DEFAULT ''",
        "ecommerce": "INTEGER NOT NULL DEFAULT 1",
        "retail": "INTEGER NOT NULL DEFAULT 1",
        "atm": "INTEGER NOT NULL DEFAULT 1",
    },
}


def _print_cards(db: sqlite3.Connection) -> None:
    """Give a number and dates to any card that predates them.

    The columns above default to empty, which is not a card number. Filling
    them is the same job as filling the categories on the bills: a migration
    is the rows as well as the columns.
    """
    rows = db.execute("SELECT last4 FROM cards WHERE number = ''").fetchall()
    if not rows:
        return
    valid_from, valid_to = _validity(datetime.now())
    with db:
        for row in rows:
            db.execute(
                "UPDATE cards SET number = ?, valid_from = ?, valid_to = ? WHERE last4 = ?",
                (_number_for(row["last4"]), valid_from, valid_to, row["last4"]),
            )


def _widen(db: sqlite3.Connection) -> None:
    """Bring an older database up to the current column set."""
    for table, columns in _ADDED.items():
        if not db.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
        ).fetchone():
            continue
        present = {row["name"] for row in db.execute(f"PRAGMA table_info({table})")}
        for column, ddl in columns.items():
            if column not in present:
                log.info("adding %s.%s to an older bank file", table, column)
                db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
    db.commit()


# A demo card number: a fixed issuer range, a random middle, and the four
# digits the card is already known by, so the two agree with each other.
_ISSUER = "220543"


def _number_for(last4: str) -> str:
    middle = f"{secrets.randbelow(1_000_000):06d}"
    return f"{_ISSUER}{middle}{last4}"


def _validity(now: datetime) -> tuple[str, str]:
    """Issued this month, good for five years, as MM/YY."""
    return now.strftime("%m/%y"), now.replace(year=now.year + 5).strftime("%m/%y")


def _card(row: sqlite3.Row) -> Card:
    return Card(
        last4=row["last4"],
        account=row["account"],
        frozen=bool(row["frozen"]),
        kind=row["kind"],
        holder=row["holder"],
        number=row["number"],
        valid_from=row["valid_from"],
        valid_to=row["valid_to"],
        ecommerce=bool(row["ecommerce"]),
        retail=bool(row["retail"]),
        atm=bool(row["atm"]),
    )


def _transaction(row: sqlite3.Row) -> Transaction:
    return Transaction(
        id=int(row["id"]),
        account=row["account"],
        merchant=row["merchant"],
        cents=int(row["cents"]),
        occurred_at=row["occurred_at"],
    )


def _handoff(row: sqlite3.Row) -> Handoff:
    return Handoff(
        reference=row["reference"],
        session_id=row["session_id"],
        turn_index=row["turn_index"],
        reason=row["reason"],
        note=row["note"],
        created_at=row["created_at"],
    )


def _stamp(moment: datetime) -> str:
    """One timestamp format, so string comparison in SQL orders correctly."""
    return moment.isoformat(sep=" ", timespec="seconds")
