"""Opening a bank for a test, whichever database this run is testing.

    pytest                                    # SQLite in memory: offline, the default
    VOICEBOT_TEST_DATABASE=postgres pytest    # the same suite against PostgreSQL

The PostgreSQL run never touches the real bank. It uses its own database,
`voicebot_test` (override with VOICEBOT_TEST_DB_NAME), creates it if it is
missing, and gives every test a schema of its own that is dropped when the
test ends -- so tests stay as isolated from each other as `:memory:` made
them. Connection settings come from VOICEBOT_DB_* or .env, as the server's do.

The few tests that look at storage directly go through `rows`, `execute` and
`columns` here rather than at a driver, so they ask the same question of
either database.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from voicebot.bank.base import Bank
from voicebot.bank.sqlite_bank import SqliteBank

TESTING_POSTGRES = os.environ.get("VOICEBOT_TEST_DATABASE", "sqlite").lower() == "postgres"


def _test_conninfo(**overrides: str) -> str:
    from voicebot.bank.postgres_bank import conninfo_from_env
    from voicebot.utils.env import load_env_file

    load_env_file(Path(__file__).resolve().parent.parent / ".env")
    name = os.environ.get("VOICEBOT_TEST_DB_NAME", "voicebot_test")
    if name == os.environ.get("VOICEBOT_DB_NAME", "voicebot"):
        raise RuntimeError(
            f"refusing to run tests in {name!r}, the database the server uses; "
            "set VOICEBOT_TEST_DB_NAME to a database of their own"
        )
    return conninfo_from_env(**{"dbname": name, **overrides})


def _ensure_test_database() -> None:
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import conninfo_to_dict

    from voicebot.bank.postgres_bank import conninfo_from_env

    name = str(conninfo_to_dict(_test_conninfo())["dbname"])
    with psycopg.connect(conninfo_from_env(dbname="postgres"), autocommit=True) as conn:
        if conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone():
            return
        conn.execute(
            sql.SQL(
                "CREATE DATABASE {} WITH ENCODING 'UTF8' "
                "LC_COLLATE 'C' LC_CTYPE 'C' TEMPLATE template0"
            ).format(sql.Identifier(name))
        )


@contextmanager
def fresh_postgres_bank(**kwargs: Any) -> Iterator[Bank]:
    """A PostgresBank in a schema made for this test and dropped after it."""
    import psycopg
    from psycopg import sql

    from voicebot.bank.postgres_bank import PostgresBank

    _ensure_test_database()
    schema = f"test_{uuid.uuid4().hex[:16]}"
    with psycopg.connect(_test_conninfo(), autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    bank = PostgresBank(_test_conninfo(options=f"-c search_path={schema}"), pool_size=4, **kwargs)
    try:
        yield bank
    finally:
        bank.close()
        with psycopg.connect(_test_conninfo(), autocommit=True) as conn:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


@contextmanager
def fresh_bank(**kwargs: Any) -> Iterator[Bank]:
    """A seeded bank of whichever kind this run is testing."""
    if TESTING_POSTGRES:
        with fresh_postgres_bank(**kwargs) as bank:
            yield bank
    else:
        bank = SqliteBank(**kwargs)
        try:
            yield bank
        finally:
            bank.close()


# --- looking at storage directly -------------------------------------------------


def rows(bank: Bank, query: str) -> list[Mapping[str, Any]]:
    """Run a read with no parameters. Rows are indexed by column name."""
    if isinstance(bank, SqliteBank):
        return [dict(row) for row in bank._db.execute(query).fetchall()]
    with bank._pool.connection() as conn:  # type: ignore[attr-defined]
        return [dict(row) for row in conn.execute(query).fetchall()]


def execute(bank: Bank, statement: str) -> None:
    """Run a write with no parameters, committed at once."""
    if isinstance(bank, SqliteBank):
        with bank._db:
            bank._db.execute(statement)
        return
    with bank._pool.connection() as conn:  # type: ignore[attr-defined]
        conn.execute(statement)


def columns(bank: Bank, table: str) -> set[str]:
    """The column names one table has, asked the way each database answers."""
    if isinstance(bank, SqliteBank):
        return {row["name"] for row in bank._db.execute(f"PRAGMA table_info({table})")}
    with bank._pool.connection() as conn:  # type: ignore[attr-defined]
        found = conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = current_schema() AND table_name = %s",
            (table,),
        ).fetchall()
    return {row["column_name"] for row in found}


__all__ = [
    "TESTING_POSTGRES",
    "columns",
    "execute",
    "fresh_bank",
    "fresh_postgres_bank",
    "rows",
]
