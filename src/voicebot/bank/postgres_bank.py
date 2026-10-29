"""The bank, backed by PostgreSQL.

The same bank as `SqliteBank` -- every method, every rule, every sentence a
customer can hear -- on a database server rather than a file. Three things
differ, each on purpose:

* **A connection pool instead of one shared connection.** Every call takes a
  connection of its own and hands it back. The SQLite bank shares a single
  connection across the server's threads, and under concurrent requests that
  interleaves their transactions: commits land in the wrong place and reads
  come back empty. A pooled connection is never used by two requests at once.

* **Two decisions are taken under a row lock.** Paying a bill and checking a
  PIN each read a value and then wrote based on it. Done in two steps, two
  requests could both read the same value: a bill paid twice, a burst of
  wrong PINs that never tripped the lockout. Here the row is locked for the
  length of the decision, so the second request waits and then sees what the
  first one did.

* **Time is a TIMESTAMP column.** It goes in as a `datetime` and comes back
  out as the same "YYYY-MM-DD HH:MM:SS" text the rest of the code has always
  been given, so nothing above the bank can tell the difference.

Connection settings come from the environment (see `conninfo_from_env`);
nothing here names a host, a user or a password.
"""

from __future__ import annotations

import os
import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from voicebot.bank.base import (
    BILL_ACCOUNT,
    CARD_CONTROLS,
    DISPUTE_REASONS,
    DISPUTE_WINDOW_DAYS,
    HANDOFF_REASONS,
    PERIODS,
    PIN_LENGTH,
    PIN_LOCK_MINUTES,
    PIN_TRIES,
    Account,
    Bank,
    BankError,
    Bill,
    Card,
    Dispute,
    Handoff,
    Payee,
    Statement,
    StorageError,
    Transaction,
    account_not_payee_message,
    bad_pin_message,
    format_money,
    period_range,
    pin_locked_message,
    too_old_to_dispute_message,
    unknown_control_message,
    unknown_dispute_reason_message,
    unknown_payee_message,
    unknown_period_message,
    unknown_reason_message,
)
from voicebot.bank.migrations import migrate
from voicebot.bank.seed import ACCOUNTS, BILLS, CARDS, HOLDER, PAYEES, SPENDING

# Shared with the SQLite bank until it is retired, so both mint card numbers,
# references and PIN hashes identically -- and so the test suite's lowered
# PIN work factor applies to both.
from voicebot.bank.sqlite_bank import DEMO_PIN, _hash, _number_for, _reference, _validity
from voicebot.utils.logging import get_logger

log = get_logger("bank.postgres")

# Where each connection setting comes from. Only what is set is passed on;
# anything left out falls back to libpq's own defaults.
ENVIRONMENT: dict[str, str] = {
    "host": "VOICEBOT_DB_HOST",
    "port": "VOICEBOT_DB_PORT",
    "dbname": "VOICEBOT_DB_NAME",
    "user": "VOICEBOT_DB_USER",
    "password": "VOICEBOT_DB_PASSWORD",
}
DEFAULTS: dict[str, str] = {
    "host": "localhost",
    "port": "5432",
    "dbname": "voicebot",
    # Names the connections in pg_stat_activity, so a DBA can see whose they are.
    "application_name": "voicebot",
    # Fail in seconds rather than hang when the server is not there.
    "connect_timeout": "5",
}


def conninfo_from_env(**overrides: str) -> str:
    """Connection settings from VOICEBOT_DB_*, as a libpq conninfo string.

    `make_conninfo` quotes each value, so a password with spaces, quotes or
    an equals sign in it arrives intact.
    """
    params = dict(DEFAULTS)
    for key, variable in ENVIRONMENT.items():
        value = os.environ.get(variable, "").strip()
        if value:
            params[key] = value
    params.update(overrides)
    return make_conninfo(**params)


def describe(conninfo: str) -> str:
    """Which database `conninfo` reaches, as user@host:port/name.

    For messages a person reads. The password is the one setting left out.
    """
    params = conninfo_to_dict(conninfo)
    return f"{params.get('user')}@{params.get('host')}:{params.get('port')}/{params.get('dbname')}"


class PostgresBank(Bank):
    """A bank in a PostgreSQL database, reached through a connection pool."""

    def __init__(
        self,
        conninfo: str,
        seed_if_empty: bool = True,
        pool_size: int = 10,
    ) -> None:
        # One plain connection first, so a wrong password or a missing
        # database fails here with the server's own words. A pool that cannot
        # connect only reports that it timed out.
        try:
            psycopg.connect(conninfo).close()
        except psycopg.Error as exc:
            raise StorageError(f"could not connect to PostgreSQL: {exc}") from exc

        # Autocommit, with explicit transaction blocks wherever two statements
        # must happen together. A lone read or write commits on its own, the
        # way it does in SQLite, and nothing is ever left half-open.
        self._pool: ConnectionPool = ConnectionPool(
            conninfo,
            min_size=1,
            max_size=pool_size,
            kwargs={"autocommit": True, "row_factory": dict_row},
            open=False,
            name="voicebot-bank",
        )
        try:
            self._pool.open(wait=True, timeout=10)
        except psycopg.Error as exc:
            raise StorageError(f"could not open a PostgreSQL connection pool: {exc}") from exc

        with self._connect() as conn:
            for version in migrate(conn):
                log.info("schema at %s", version)
            empty = conn.execute("SELECT 1 FROM accounts LIMIT 1").fetchone() is None
            if seed_if_empty and empty:
                with conn.transaction():
                    _seed(conn, datetime.now())
                    _print_cards(conn)
                log.info("seeded a new bank")
            elif seed_if_empty:
                # The one repair the SQLite bank makes on open that can still
                # apply here: a card with no number or no PIN. The others
                # filled tables and columns an older SQLite file lacked, and
                # a database built by the migrations has never lacked them.
                _print_cards(conn)

    def close(self) -> None:
        self._pool.close()

    # --- reads ---------------------------------------------------------------

    def accounts(self) -> tuple[Account, ...]:
        with self._connect() as conn:
            rows = conn.execute("SELECT name, balance_cents FROM accounts ORDER BY name").fetchall()
        return tuple(Account(name=r["name"], balance_cents=int(r["balance_cents"])) for r in rows)

    def balance(self, account: str) -> int:
        with self._connect() as conn:
            return self._balance(conn, self._account(conn, account))

    def cards(self) -> tuple[Card, ...]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM cards ORDER BY last4").fetchall()
        return tuple(_card(row) for row in rows)

    def bills(self) -> tuple[Bill, ...]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM bills ORDER BY due_day, name").fetchall()
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
        with self._connect() as conn:
            rows = conn.execute("SELECT name FROM payees ORDER BY name").fetchall()
        return tuple(Payee(name=r["name"]) for r in rows)

    def transactions(
        self,
        limit: int = 5,
        account: str | None = None,
        merchant: str | None = None,
        period: str | None = None,
    ) -> tuple[Transaction, ...]:
        clauses: list[sql.Composable] = []
        params: list[object] = []
        with self._connect() as conn:
            if account:
                clauses.append(sql.SQL("account = %s"))
                params.append(self._account(conn, account))
            if merchant:
                clauses.append(sql.SQL("merchant = %s"))
                params.append(merchant.lower())
            if period is not None:
                # The same half-open range `spending` totals over, so a list
                # and a total of it cannot disagree about where a period ends.
                if period not in PERIODS:
                    raise BankError(unknown_period_message(period))
                start, end = period_range(period, datetime.now())
                clauses.append(sql.SQL("occurred_at >= %s AND occurred_at < %s"))
                params.extend([start, end])

            query: sql.Composable = sql.SQL(
                "SELECT id, account, merchant, cents, occurred_at FROM transactions"
            )
            if clauses:
                query += sql.SQL(" WHERE ") + sql.SQL(" AND ").join(clauses)
            # id breaks ties: the seed writes several rows inside one second.
            query += sql.SQL(" ORDER BY occurred_at DESC, id DESC LIMIT %s")
            params.append(max(1, limit))
            rows = conn.execute(query, params).fetchall()
        return tuple(_transaction(row) for row in rows)

    def spending(self, period: str, merchant: str | None = None) -> int:
        if period not in PERIODS:
            raise BankError(unknown_period_message(period))
        start, end = period_range(period, datetime.now())

        query = (
            "SELECT COALESCE(SUM(-cents), 0) AS total FROM transactions "
            "WHERE cents < 0 AND occurred_at >= %s AND occurred_at < %s"
        )
        params: list[object] = [start, end]
        if merchant:
            query += " AND merchant = %s"
            params.append(merchant.lower())

        with self._connect() as conn:
            row = _one(conn.execute(query, params))
        # SUM over a BIGINT is NUMERIC in PostgreSQL; the total is whole cents.
        return int(row["total"])

    # --- writes --------------------------------------------------------------

    def transfer(self, cents: int, payee: str, from_account: str) -> int:
        if cents <= 0:
            raise BankError("The amount has to be more than zero.")
        with self._connect() as conn:
            account = self._account(conn, from_account)
            payee = self._payee(conn, payee)
            # The sufficiency test is the UPDATE's own WHERE clause, so two
            # transfers at once cannot both pass a check made beforehand.
            with conn.transaction():
                changed = conn.execute(
                    "UPDATE accounts SET balance_cents = balance_cents - %s "
                    "WHERE name = %s AND balance_cents >= %s",
                    (cents, account, cents),
                ).rowcount
                if not changed:
                    available = self._balance(conn, account)
                    raise BankError(
                        f"There isn't enough in {account}. "
                        f"The balance is {format_money(available)}."
                    )
                self._ledger(conn, account, payee.lower(), -cents)
            return self._balance(conn, account)

    def move(self, cents: int, from_account: str, to_account: str) -> tuple[int, int]:
        if cents <= 0:
            raise BankError("The amount has to be more than zero.")
        with self._connect() as conn:
            source = self._account(conn, from_account)
            target = self._account(conn, to_account)
            if source == target:
                raise BankError(f"{source} and {target} are the same account.")

            # Both sides in one transaction: there is no moment where the
            # money has left one account and not arrived in the other.
            with conn.transaction():
                changed = conn.execute(
                    "UPDATE accounts SET balance_cents = balance_cents - %s "
                    "WHERE name = %s AND balance_cents >= %s",
                    (cents, source, cents),
                ).rowcount
                if not changed:
                    available = self._balance(conn, source)
                    raise BankError(
                        f"There isn't enough in {source}. "
                        f"The balance is {format_money(available)}."
                    )
                conn.execute(
                    "UPDATE accounts SET balance_cents = balance_cents + %s WHERE name = %s",
                    (cents, target),
                )
                self._ledger(conn, source, target, -cents)
                self._ledger(conn, target, source, cents)
            return self._balance(conn, source), self._balance(conn, target)

    def pay_bill(self, bill: str) -> int:
        name = bill.lower()
        with self._connect() as conn, conn.transaction():
            # Locked until this transaction ends. A second payment of the
            # same bill waits here, then finds it paid. Read and written in
            # separate steps -- as the SQLite bank does -- two payments could
            # both find it unpaid, and checking was charged twice.
            row = conn.execute(
                "SELECT amount_cents, paid FROM bills WHERE name = %s FOR UPDATE", (name,)
            ).fetchone()
            if row is None:
                known = ", ".join(
                    r["name"] for r in conn.execute("SELECT name FROM bills ORDER BY due_day, name")
                )
                raise BankError(f"I don't have a {bill} bill on file. I have: {known}.")
            if row["paid"]:
                raise BankError(f"Your {bill} bill is already paid.")

            amount = int(row["amount_cents"])
            changed = conn.execute(
                "UPDATE accounts SET balance_cents = balance_cents - %s "
                "WHERE name = %s AND balance_cents >= %s",
                (amount, BILL_ACCOUNT, amount),
            ).rowcount
            if not changed:
                available = self._balance(conn, BILL_ACCOUNT)
                raise BankError(
                    f"There isn't enough in checking to cover that. The bill is "
                    f"{format_money(amount)} and the balance is {format_money(available)}."
                )
            conn.execute("UPDATE bills SET paid = TRUE WHERE name = %s", (name,))
            self._ledger(conn, BILL_ACCOUNT, name, -amount)
        return amount

    # --- the PIN -------------------------------------------------------------

    def check_pin(self, last4: str, pin: str, now: datetime | None = None) -> bool:
        moment = now or datetime.now()
        with self._connect() as conn, conn.transaction():
            # The card is locked while this PIN is judged and the outcome
            # written. Without the lock, a burst of guesses all read the same
            # failure count, each wrote back count + 1, and the lockout never
            # tripped: twenty guesses got through a three-guess limit. With
            # it, the burst is judged one guess at a time, and the fourth
            # finds the card locked without its PIN ever being hashed.
            row = self._card_row(conn, last4, lock=True)
            if _locked(row, moment):
                # Not answered with "wrong": telling a locked card apart from
                # a wrong PIN would hand back one more guess per lock.
                raise BankError(pin_locked_message())

            if secrets.compare_digest(_hash(pin, row["pin_salt"]), row["pin_hash"]):
                conn.execute(
                    "UPDATE cards SET pin_failures = 0, pin_locked_until = NULL WHERE last4 = %s",
                    (row["last4"],),
                )
                return True

            failures = int(row["pin_failures"]) + 1
            until = None
            if failures >= PIN_TRIES:
                until = _second(moment + timedelta(minutes=PIN_LOCK_MINUTES))
                log.warning("pin locked on card %s after %d failures", row["last4"], failures)
            conn.execute(
                "UPDATE cards SET pin_failures = %s, pin_locked_until = %s WHERE last4 = %s",
                (failures, until, row["last4"]),
            )
            return False

    def pin_locked(self, last4: str, now: datetime | None = None) -> bool:
        with self._connect() as conn:
            return _locked(self._card_row(conn, last4), now or datetime.now())

    def set_pin(self, last4: str, current: str, new: str) -> None:
        if len(new) != PIN_LENGTH or not new.isdigit():
            raise BankError(bad_pin_message())
        if not self.check_pin(last4, current):
            raise BankError("That is not your current PIN.")

        salt = secrets.token_hex(16)
        with self._connect() as conn:
            card = self._card_row(conn, last4)
            conn.execute(
                "UPDATE cards SET pin_hash = %s, pin_salt = %s, pin_failures = 0, "
                "pin_locked_until = NULL WHERE last4 = %s",
                (_hash(new, salt), salt, card["last4"]),
            )

    def set_card_control(self, last4: str, control: str, enabled: bool) -> Card:
        name = control.strip().lower()
        if name not in CARD_CONTROLS:
            raise BankError(unknown_control_message(control))
        with self._connect() as conn:
            card = self._card_row(conn, last4)
            # The column comes from CARD_CONTROLS and nowhere else, and is
            # quoted as an identifier besides.
            conn.execute(
                sql.SQL("UPDATE cards SET {} = %s WHERE last4 = %s").format(sql.Identifier(name)),
                (bool(enabled), card["last4"]),
            )
            return _card(self._card_row(conn, last4))

    def freeze_card(self, last4: str) -> Card:
        with self._connect() as conn:
            changed = conn.execute(
                "UPDATE cards SET frozen = TRUE WHERE last4 = %s", (last4,)
            ).rowcount
            if not changed:
                known = ", ".join(
                    r["last4"] for r in conn.execute("SELECT last4 FROM cards ORDER BY last4")
                )
                raise BankError(f"I can't find a card ending {last4}. I have: {known}.")
            row = _one(conn.execute("SELECT * FROM cards WHERE last4 = %s", (last4,)))
        return _card(row)

    def order_card(self, account: str) -> Card:
        """Issue a new card and return it, so the bot can say which is coming."""
        with self._connect() as conn:
            name = self._account(conn, account)
            existing = conn.execute("SELECT last4, holder FROM cards ORDER BY last4").fetchall()
            taken = {row["last4"] for row in existing}
            last4 = f"{secrets.randbelow(10_000):04d}"
            while last4 in taken:
                last4 = f"{secrets.randbelow(10_000):04d}"
            holder = next((row["holder"] for row in existing if row["holder"]), "")
            valid_from, valid_to = _validity(datetime.now())
            # A replacement is a new card for the same person, so the PIN
            # travels with them, copied as the stored hash and salt. Ordered,
            # where SQLite took whichever row it met first, so the choice is
            # the same every time.
            carried = conn.execute(
                "SELECT pin_hash, pin_salt FROM cards WHERE pin_hash != '' ORDER BY last4 LIMIT 1"
            ).fetchone()
            conn.execute(
                "INSERT INTO cards (last4, account, frozen, kind, holder, "
                "number, valid_from, valid_to, pin_hash, pin_salt) "
                "VALUES (%s, %s, FALSE, 'debit', %s, %s, %s, %s, %s, %s)",
                (
                    last4,
                    name,
                    holder,
                    _number_for(last4),
                    valid_from,
                    valid_to,
                    carried["pin_hash"] if carried else "",
                    carried["pin_salt"] if carried else "",
                ),
            )
            return _card(self._card_row(conn, last4))

    def deposit(self, cents: int, account: str, source: str = "deposit") -> int:
        """Pay money in. Writes a positive ledger row, so the history stays true."""
        if cents <= 0:
            raise BankError("A deposit has to be more than zero.")
        with self._connect() as conn:
            name = self._account(conn, account)
            with conn.transaction():
                conn.execute(
                    "UPDATE accounts SET balance_cents = balance_cents + %s WHERE name = %s",
                    (cents, name),
                )
                self._ledger(conn, name, source.lower(), cents)
            return self._balance(conn, name)

    # --- audit ---------------------------------------------------------------

    def statement(self, year: int, month: int) -> Statement:
        if not 1 <= month <= 12:
            raise BankError(f"{month} is not a month.")

        rows: tuple[Transaction, ...] = ()
        # A year no calendar date can hold matches nothing, as a text prefix
        # match on it did in SQLite: an empty statement, not an error.
        if 1 <= year <= 9998:
            start = datetime(year, month, 1)
            end = datetime(year + month // 12, month % 12 + 1, 1)
            with self._connect() as conn:
                rows = tuple(
                    _transaction(row)
                    for row in conn.execute(
                        "SELECT id, account, merchant, cents, occurred_at FROM transactions "
                        "WHERE occurred_at >= %s AND occurred_at < %s "
                        "ORDER BY occurred_at DESC, id DESC",
                        (start, end),
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
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, account, merchant, cents, occurred_at FROM transactions WHERE id = %s",
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

        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, merchant, cents, occurred_at FROM transactions WHERE id = %s",
                (transaction_id,),
            ).fetchone()
            if row is None:
                raise BankError("I can't find that charge on your account.")
            if row["cents"] >= 0:
                raise BankError(
                    f"That is money paid in, not a charge. "
                    f"There is nothing to dispute about {row['merchant']}."
                )

            age = (now or datetime.now()) - row["occurred_at"]
            if age.days > DISPUTE_WINDOW_DAYS:
                raise BankError(too_old_to_dispute_message(age.days))

            already = BankError(
                f"The {row['merchant']} charge for "
                f"{format_money(abs(int(row['cents'])))} has already been disputed."
            )
            if conn.execute(
                "SELECT 1 FROM disputes WHERE transaction_id = %s", (transaction_id,)
            ).fetchone():
                raise already

            reference = _reference()
            try:
                conn.execute(
                    "INSERT INTO disputes (reference, transaction_id, merchant, "
                    "amount_cents, occurred_at, reason, note) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)",
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
            except psycopg.errors.UniqueViolation as exc:
                # Two disputes of one charge at once: both passed the check
                # above, and the unique constraint stopped the second. It is
                # the same refusal, and is worded as one.
                if exc.diag.constraint_name == "disputes_transaction_id_key":
                    raise already from exc
                raise
            return self._read_dispute(conn, reference)

    def disputes(self) -> tuple[Dispute, ...]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM disputes ORDER BY created_at DESC, seq DESC"
            ).fetchall()
        return tuple(_dispute(row) for row in rows)

    def _read_dispute(self, conn: psycopg.Connection[Any], reference: str) -> Dispute:
        """Read one case back. No join: the charge is held on the case."""
        row = _one(conn.execute("SELECT * FROM disputes WHERE reference = %s", (reference,)))
        return _dispute(row)

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

        with self._connect() as conn:
            row = _one(
                conn.execute(
                    "INSERT INTO handoffs (reference, session_id, turn_index, reason, note) "
                    "VALUES (%s, %s, %s, %s, %s) RETURNING *",
                    (_reference(), session_id, turn_index, key, note),
                )
            )
        return _handoff(row)

    def handoffs(self, session_id: str | None = None) -> tuple[Handoff, ...]:
        query = "SELECT * FROM handoffs"
        params: list[object] = []
        if session_id:
            query += " WHERE session_id = %s"
            params.append(session_id)
        query += " ORDER BY created_at DESC, seq DESC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return tuple(_handoff(row) for row in rows)

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
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO turns (session_id, turn_index, user_text, bot_text, "
                "tool_calls, guard_hit, suppressed, latency_ms) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    session_id,
                    turn_index,
                    user_text,
                    bot_text,
                    tool_calls,
                    bool(guard_hit),
                    suppressed,
                    latency_ms,
                ),
            )

    def turn_count(self) -> int:
        with self._connect() as conn:
            return int(_one(conn.execute("SELECT COUNT(*) AS n FROM turns"))["n"])

    def guard_hit_count(self) -> int:
        with self._connect() as conn:
            return int(_one(conn.execute("SELECT COUNT(*) AS n FROM turns WHERE guard_hit"))["n"])

    def reset(self) -> None:
        """Back to the opening state. `turns` survives -- it is the record.

        One transaction, so a reset that fails half way leaves the bank as it
        was rather than empty. The ledger's numbering starts again at 1, as
        it does in SQLite, and the cards are printed as well as seeded.
        """
        with self._connect() as conn, conn.transaction():
            for table in ("transactions", "cards", "bills", "accounts", "payees"):
                conn.execute(sql.SQL("DELETE FROM {}").format(sql.Identifier(table)))
            conn.execute("SELECT setval(pg_get_serial_sequence('transactions', 'id'), 1, false)")
            _seed(conn, datetime.now())
            _print_cards(conn)
        log.info("bank reset to seeded state")

    # --- internals -----------------------------------------------------------

    @contextmanager
    def _connect(self) -> Iterator[psycopg.Connection[Any]]:
        """A pooled connection, with the driver's failures turned into ours."""
        try:
            with self._pool.connection() as conn:
                yield conn
        except psycopg.Error as exc:
            raise StorageError(str(exc)) from exc

    def _account(self, conn: psycopg.Connection[Any], name: str) -> str:
        """Validate an account name, or say what exists."""
        key = name.strip().lower()
        if conn.execute("SELECT 1 FROM accounts WHERE name = %s", (key,)).fetchone():
            return key
        known = ", ".join(
            r["name"] for r in conn.execute("SELECT name FROM accounts ORDER BY name")
        )
        raise BankError(f"I don't recognise the account {name!r}. I have: {known}.")

    def _payee(self, conn: psycopg.Connection[Any], name: str) -> str:
        """Validate a payee, or say plainly why this one will not do."""
        key = name.strip().lower()
        if conn.execute("SELECT 1 FROM payees WHERE name = %s", (key,)).fetchone():
            return key
        if conn.execute("SELECT 1 FROM accounts WHERE name = %s", (key,)).fetchone():
            raise BankError(account_not_payee_message(key))
        known = tuple(r["name"] for r in conn.execute("SELECT name FROM payees ORDER BY name"))
        raise BankError(unknown_payee_message(name, known))

    def _balance(self, conn: psycopg.Connection[Any], account: str) -> int:
        row = conn.execute(
            "SELECT balance_cents FROM accounts WHERE name = %s", (account,)
        ).fetchone()
        assert row is not None  # every caller has validated the account
        return int(row["balance_cents"])

    def _card_row(
        self, conn: psycopg.Connection[Any], last4: str, lock: bool = False
    ) -> dict[str, Any]:
        """One card, or a sentence naming the ones that exist."""
        key = last4.strip()
        query = "SELECT * FROM cards WHERE last4 = %s"
        if lock:
            query += " FOR UPDATE"
        row = conn.execute(query, (key,)).fetchone()
        if row is None:
            known = ", ".join(
                r["last4"] for r in conn.execute("SELECT last4 FROM cards ORDER BY last4")
            )
            raise BankError(f"I don't have a card ending {last4}. I have: {known}.")
        return row

    def _ledger(
        self, conn: psycopg.Connection[Any], account: str, merchant: str, cents: int
    ) -> None:
        """Record a movement. Called inside an open transaction."""
        conn.execute(
            "INSERT INTO transactions (account, merchant, cents, occurred_at) "
            "VALUES (%s, %s, %s, %s)",
            (account, merchant, cents, _second(datetime.now())),
        )


# --- seeding and printing ------------------------------------------------------


def _seeded_at(period: str, now: datetime) -> datetime:
    """A moment inside `period` that is never in the future, as the seed uses."""
    start, _ = period_range(period, now)
    return _second(min(start + timedelta(hours=1), now))


def _seed(conn: psycopg.Connection[Any], now: datetime) -> None:
    """Fill an empty bank from the same constants the SQLite seed uses."""
    with conn.cursor() as cur:
        if cur.execute("SELECT 1 FROM payees LIMIT 1").fetchone() is None:
            cur.executemany("INSERT INTO payees (name) VALUES (%s)", [(name,) for name in PAYEES])
        cur.executemany("INSERT INTO accounts (name, balance_cents) VALUES (%s, %s)", ACCOUNTS)
        cur.executemany(
            "INSERT INTO cards (last4, account, frozen, kind, holder) "
            "VALUES (%s, %s, FALSE, %s, %s)",
            [(last4, account, kind, HOLDER) for last4, account, kind in CARDS],
        )
        cur.executemany(
            "INSERT INTO bills (name, amount_cents, paid, category, due_day) "
            "VALUES (%s, %s, FALSE, %s, %s)",
            BILLS,
        )
        cur.executemany(
            "INSERT INTO transactions (account, merchant, cents, occurred_at) "
            "VALUES ('checking', %s, %s, %s)",
            [(merchant, cents, _seeded_at(period, now)) for merchant, cents, period in SPENDING],
        )


def _print_cards(conn: psycopg.Connection[Any]) -> None:
    """Give a name, a number, dates and a PIN to any card that lacks them.

    Every card gets its own salt even though the demo PIN is the same, so two
    hashes never match and one says nothing about the other.
    """
    rows = conn.execute("SELECT last4, holder, number, pin_hash FROM cards").fetchall()
    valid_from, valid_to = _validity(datetime.now())
    with conn.transaction():
        for row in rows:
            if not row["holder"]:
                conn.execute(
                    "UPDATE cards SET holder = %s WHERE last4 = %s", (HOLDER, row["last4"])
                )
            if not row["number"]:
                conn.execute(
                    "UPDATE cards SET number = %s, valid_from = %s, valid_to = %s WHERE last4 = %s",
                    (_number_for(row["last4"]), valid_from, valid_to, row["last4"]),
                )
            if not row["pin_hash"]:
                salt = secrets.token_hex(16)
                conn.execute(
                    "UPDATE cards SET pin_hash = %s, pin_salt = %s WHERE last4 = %s",
                    (_hash(DEMO_PIN, salt), salt, row["last4"]),
                )


# --- converting rows -----------------------------------------------------------


def _one(cursor: psycopg.Cursor[Any]) -> dict[str, Any]:
    """The single row a statement is known to return: an aggregate, or a
    row this same call just wrote. If it is ever missing, the database
    broke a promise, and that is a storage failure rather than a refusal."""
    row = cursor.fetchone()
    if row is None:
        raise StorageError("expected a row from the database and got none")
    return dict(row)


def _second(moment: datetime) -> datetime:
    """Whole seconds, as SQLite's text stamps were: the columns are TIMESTAMP(0)."""
    return moment.replace(microsecond=0)


def _text(moment: datetime) -> str:
    """A timestamp as the rest of the code has always received it."""
    return moment.strftime("%Y-%m-%d %H:%M:%S")


def _locked(row: dict[str, Any], now: datetime) -> bool:
    until = row["pin_locked_until"]
    return until is not None and _second(now) < until


def _card(row: dict[str, Any]) -> Card:
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


def _transaction(row: dict[str, Any]) -> Transaction:
    return Transaction(
        id=int(row["id"]),
        account=row["account"],
        merchant=row["merchant"],
        cents=int(row["cents"]),
        occurred_at=_text(row["occurred_at"]),
    )


def _dispute(row: dict[str, Any]) -> Dispute:
    return Dispute(
        reference=row["reference"],
        transaction_id=int(row["transaction_id"]),
        merchant=row["merchant"],
        amount_cents=int(row["amount_cents"]),
        occurred_at=_text(row["occurred_at"]),
        reason=row["reason"],
        note=row["note"],
        status=row["status"],
        created_at=_text(row["created_at"]),
    )


def _handoff(row: dict[str, Any]) -> Handoff:
    return Handoff(
        reference=row["reference"],
        session_id=row["session_id"],
        turn_index=row["turn_index"],
        reason=row["reason"],
        note=row["note"],
        created_at=_text(row["created_at"]),
    )
