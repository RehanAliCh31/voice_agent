"""The bank: accounts, cards, bills, a ledger and an audit log."""

from __future__ import annotations

from typing import TYPE_CHECKING

from voicebot.bank.base import (
    PERIODS,
    Account,
    Bank,
    BankError,
    Bill,
    Card,
    format_money,
    period_range,
)
from voicebot.bank.sqlite_bank import SqliteBank

if TYPE_CHECKING:
    from voicebot.config import DatabaseConfig

__all__ = [
    "PERIODS",
    "Account",
    "Bank",
    "BankError",
    "Bill",
    "Card",
    "SqliteBank",
    "format_money",
    "open_bank",
    "period_range",
]


def open_bank(database: DatabaseConfig) -> Bank:
    """The bank `database` names, open and ready.

    The one place that knows which class each engine means, so the server and
    the shell commands that look after the bank can never open different ones.
    """
    if database.engine == "postgres":
        # Imported here so that psycopg is needed only by a bank that uses it:
        # the offline path and the default test run never load it.
        from voicebot.bank.postgres_bank import PostgresBank, conninfo_from_env

        return PostgresBank(
            conninfo_from_env(),
            seed_if_empty=database.seed_on_create,
            pool_size=database.pool_size,
            pool_timeout_s=database.pool_timeout_s,
            statement_timeout_ms=database.statement_timeout_ms,
            lock_timeout_ms=database.lock_timeout_ms,
        )
    if database.engine == "sqlite":
        return SqliteBank(path=database.path, seed_if_empty=database.seed_on_create)
    raise ValueError(f"unknown database engine {database.engine!r}. Known: postgres, sqlite.")
