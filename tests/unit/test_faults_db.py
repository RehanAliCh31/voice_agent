"""Fault injection: the database hanging, full, dropped, or raced.

The PostgreSQL tests run against a real server, in a schema of their own,
when VOICEBOT_TEST_DATABASE=postgres; the SQLite ones always run. What they
check is that no wait is unbounded and that a failure arrives as a
StorageError the rest of the bot knows how to say -- never a hang, and never
a payment made twice.
"""

from __future__ import annotations

import sqlite3
import threading
import time

import pytest

from tests import support
from voicebot.bank.base import BankError, StorageError
from voicebot.bank.sqlite_bank import SqliteBank

postgres = pytest.mark.skipif(not support.TESTING_POSTGRES, reason="needs PostgreSQL")


# --- PostgreSQL ------------------------------------------------------------------


@postgres
def test_a_statement_that_runs_too_long_is_cut_off() -> None:
    with support.fresh_postgres_bank(statement_timeout_ms=200) as bank:
        started = time.monotonic()
        with pytest.raises(StorageError, match="statement timeout"), bank._connect() as conn:  # type: ignore[attr-defined]
            conn.execute("SELECT pg_sleep(5)")
        assert time.monotonic() - started < 2.0


@postgres
def test_no_free_connection_is_refused_quickly_rather_than_queued() -> None:
    with support.fresh_postgres_bank(pool_size=1, pool_timeout_s=0.3) as bank:
        held = threading.Event()
        release = threading.Event()

        def hold() -> None:
            with bank._connect():  # type: ignore[attr-defined]
                held.set()
                release.wait(5)

        holder = threading.Thread(target=hold)
        holder.start()
        held.wait(5)
        started = time.monotonic()
        try:
            with pytest.raises(StorageError):
                bank.accounts()
            assert time.monotonic() - started < 2.0
        finally:
            release.set()
            holder.join(5)
        # And once it is free again, it is used.
        assert bank.accounts()


@postgres
def test_a_connection_the_server_dropped_is_replaced_before_it_is_used() -> None:
    import psycopg

    with support.fresh_postgres_bank(pool_size=1) as bank:
        with bank._connect() as conn:  # type: ignore[attr-defined]
            pid = conn.execute("SELECT pg_backend_pid() AS pid").fetchone()["pid"]
        with psycopg.connect(support._test_conninfo(), autocommit=True) as admin:
            admin.execute("SELECT pg_terminate_backend(%s)", (pid,))
        # The pool checks it on the way out, finds it dead, and hands out a new one.
        assert bank.accounts()


@postgres
def test_a_failed_start_hands_its_connections_back(monkeypatch: pytest.MonkeyPatch) -> None:
    from voicebot.bank import postgres_bank

    closed: list[bool] = []
    original = postgres_bank.ConnectionPool.close

    def recording_close(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        closed.append(True)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(postgres_bank.ConnectionPool, "close", recording_close)
    monkeypatch.setattr(
        postgres_bank.PostgresBank,
        "_prepare",
        lambda self, seed: (_ for _ in ()).throw(StorageError("migration failed")),
    )
    with pytest.raises(StorageError), support.fresh_postgres_bank():
        pass
    assert closed, "the pool was closed rather than left open"


# --- SQLite ------------------------------------------------------------------------


def test_a_bill_paid_from_two_threads_at_once_is_paid_once() -> None:
    bank = SqliteBank()
    before = bank.balance("checking")
    results: list[str] = []
    start = threading.Barrier(4)

    def pay() -> None:
        start.wait()
        try:
            bank.pay_bill("internet")
            results.append("paid")
        except BankError:
            results.append("refused")

    threads = [threading.Thread(target=pay) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)

    assert results.count("paid") == 1
    assert bank.balance("checking") == before - 4_000


def test_a_driver_error_arrives_as_a_storage_error() -> None:
    bank = SqliteBank()
    bank._db.close()  # the file went away, say
    with pytest.raises(StorageError) as raised:
        bank.accounts()
    assert isinstance(raised.value.__cause__, sqlite3.Error)


def test_the_bank_answers_a_readiness_ping() -> None:
    SqliteBank().ping()


def test_a_bank_opened_on_the_29th_of_february_still_opens() -> None:
    from datetime import datetime

    from voicebot.bank.sqlite_bank import _validity

    assert _validity(datetime(2028, 2, 29)) == ("02/28", "02/33")
    assert _validity(datetime(2026, 10, 30)) == ("10/26", "10/31")
