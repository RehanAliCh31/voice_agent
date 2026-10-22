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

from voicebot.bank.base import period_range

ACCOUNTS: tuple[tuple[str, int], ...] = (
    ("checking", 248_350),
    ("savings", 1_902_000),
)

CARDS: tuple[tuple[str, str], ...] = (
    ("4417", "checking"),
    ("9021", "savings"),
)

BILLS: tuple[tuple[str, int], ...] = (
    ("electricity", 7_250),
    ("water", 3_100),
    ("phone", 2_499),
    ("internet", 4_000),
    ("credit card", 21_500),
)

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
    with connection:
        connection.executemany("INSERT INTO accounts VALUES (?, ?)", ACCOUNTS)
        connection.executemany("INSERT INTO cards VALUES (?, ?, 0)", CARDS)
        connection.executemany("INSERT INTO bills VALUES (?, ?, 0)", BILLS)
        connection.executemany(
            "INSERT INTO transactions (account, merchant, cents, occurred_at) "
            "VALUES ('checking', ?, ?, ?)",
            [(merchant, cents, _occurred_at(period, now)) for merchant, cents, period in SPENDING],
        )


def clear(connection: sqlite3.Connection) -> None:
    """Empty the account tables, leaving `turns` untouched."""
    with connection:
        for table in ("transactions", "cards", "bills", "accounts"):
            connection.execute(f"DELETE FROM {table}")
        connection.execute("DELETE FROM sqlite_sequence WHERE name = 'transactions'")


def main() -> None:
    """Create or reseed a database file from the shell."""
    from voicebot.bank.sqlite_bank import SqliteBank

    parser = argparse.ArgumentParser(description="Create or reseed the demo bank.")
    parser.add_argument("--path", default="data/bank.db", help="database file to write")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="reseed an existing database, keeping its audit log",
    )
    args = parser.parse_args()

    bank = SqliteBank(path=args.path, seed_if_empty=not args.reset)
    if args.reset:
        bank.reset()
    print(f"{'reseeded' if args.reset else 'created'} {args.path}")


if __name__ == "__main__":
    main()
