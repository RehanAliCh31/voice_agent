"""The demo bank's opening state.

Dates are computed relative to the moment of seeding rather than hard-coded, so
"what did I spend this week" keeps working next month. Each transaction is
placed just after the start of the period it is meant to fall in, which is
always in the past whatever day the demo runs on.
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from voicebot.bank.base import format_money, period_range

if TYPE_CHECKING:
    from voicebot.bank.sqlite_bank import SqliteBank

ACCOUNTS: tuple[tuple[str, int], ...] = (
    ("checking", 248_350),
    ("savings", 1_902_000),
)

# The name printed on the card. One customer, so one name.
HOLDER = "REHAN ALI CH"

# (last four, account it draws on, what kind it is). The credit card is the
# one the "credit card" bill belongs to, so the two agree with each other.
CARDS: tuple[tuple[str, str, str], ...] = (
    ("4417", "checking", "debit"),
    ("9021", "savings", "credit"),
)

# (name, cents, category, day of the month it falls due). Everything that
# leaves every month, not only the things called bills -- the monthly screen
# shows all of it, and pay_bill works on all of it.
#
# Due days stop at 28 so that every month has one. A bill due on the 31st
# has no February.
BILLS: tuple[tuple[str, int, str, int], ...] = (
    ("rent", 120_000, "rent", 1),
    ("groceries", 45_000, "groceries", 1),
    ("electricity", 7_250, "bills", 5),
    ("spotify", 999, "subscriptions", 7),
    ("internet", 4_000, "wifi", 8),
    ("water", 3_100, "bills", 12),
    ("phone", 2_499, "bills", 18),
    ("netflix", 1_099, "subscriptions", 19),
    ("credit card", 21_500, "bills", 22),
)

# Who the customer can send money to. Names only: this bank's payees are
# people, and an account name is deliberately not among them.
PAYEES: tuple[str, ...] = ("sarah", "ali")

# (merchant, cents, the period it should fall in). Cents are negative because
# this is money leaving the account.
SPENDING: tuple[tuple[str, int, str], ...] = (
    ("amazon", -8_499, "last month"),
    ("tesco", -5_210, "last month"),
    ("netflix", -1_099, "this week"),
    ("uber", -2_340, "this week"),
    ("starbucks", -760, "yesterday"),
)


def _occurred_at(period: str, now: datetime) -> str:
    """A timestamp inside `period` that is never in the future."""
    start, _ = period_range(period, now)
    return min(start + timedelta(hours=1), now).isoformat(sep=" ", timespec="seconds")


def seed(connection: sqlite3.Connection, now: datetime | None = None) -> None:
    """Fill an empty database. Assumes the schema already exists."""
    now = now or datetime.now()
    seed_payees(connection)
    with connection:
        connection.executemany("INSERT INTO accounts VALUES (?, ?)", ACCOUNTS)
        connection.executemany(
            "INSERT INTO cards (last4, account, frozen, kind, holder) " "VALUES (?, ?, 0, ?, ?)",
            [(last4, account, kind, HOLDER) for last4, account, kind in CARDS],
        )
        connection.executemany(
            "INSERT INTO bills (name, amount_cents, paid, category, due_day) "
            "VALUES (?, ?, 0, ?, ?)",
            BILLS,
        )
        connection.executemany(
            "INSERT INTO transactions (account, merchant, cents, occurred_at) "
            "VALUES ('checking', ?, ?, ?)",
            [(merchant, cents, _occurred_at(period, now)) for merchant, cents, period in SPENDING],
        )


def seed_payees(connection: sqlite3.Connection) -> None:
    """Fill the payee list if it is empty.

    Separate from `seed` and safe to call on a database that already holds
    accounts. A bank file created before payees existed gets the table (the
    schema runs on every open) but no rows, and every transfer would then be
    refused for want of a payee that was never inserted.
    """
    with connection:
        if connection.execute("SELECT 1 FROM payees").fetchone():
            return
        connection.executemany("INSERT INTO payees VALUES (?)", [(name,) for name in PAYEES])


def clear(connection: sqlite3.Connection) -> None:
    """Empty the account tables, leaving `turns` untouched."""
    with connection:
        for table in ("transactions", "cards", "bills", "accounts", "payees"):
            connection.execute(f"DELETE FROM {table}")
        connection.execute("DELETE FROM sqlite_sequence WHERE name = 'transactions'")


def _show(bank: SqliteBank) -> None:
    print()
    for account in bank.accounts():
        print(f"  {account.name:<10} {format_money(account.balance_cents):>14}")
    print(f"\n  {bank.turn_count()} turns in the audit log\n")


def main() -> None:
    """Look after the demo bank from the shell.

    Editing the opening balances in this file does nothing to a database that
    already exists -- the seed runs once, when the tables are empty. These are
    the commands that act on a live one.
    """
    from voicebot.bank.sqlite_bank import SqliteBank

    parser = argparse.ArgumentParser(
        description="Create, inspect, top up or reset the demo bank.",
        epilog=(
            "examples:\n"
            "  python -m voicebot.bank.seed --show\n"
            "  python -m voicebot.bank.seed --deposit 500 --account checking\n"
            "  python -m voicebot.bank.seed --reset\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--path", default="data/bank.db", help="database file to act on")
    parser.add_argument("--show", action="store_true", help="print the balances and exit")
    parser.add_argument(
        "--deposit",
        type=float,
        metavar="DOLLARS",
        help="pay money into an account, keeping everything else as it is",
    )
    parser.add_argument(
        "--account", default="checking", help="which account to pay into (default: checking)"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="restore the opening balances, keeping the audit log",
    )
    args = parser.parse_args()

    existed = Path(args.path).exists() if args.path != ":memory:" else False
    bank = SqliteBank(path=args.path, seed_if_empty=True)
    if not existed:
        print(f"created {args.path}")

    if args.reset:
        bank.reset()
        print(f"reset {args.path} to its opening balances")
    elif args.deposit is not None:
        # Dollars on the command line because a person types dollars; cents
        # everywhere inside, because money is integer cents throughout.
        cents = round(args.deposit * 100)
        new_balance = bank.deposit(cents, args.account, source="manual top-up")
        print(
            f"paid {format_money(cents)} into {args.account}; "
            f"it now holds {format_money(new_balance)}"
        )

    _show(bank)


if __name__ == "__main__":
    main()
