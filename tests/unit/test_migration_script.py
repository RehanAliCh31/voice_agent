"""The migration script can say no.

Its PASS is only worth something if it would have said FAIL: a verifier
that cannot tell two databases apart proves nothing by agreeing with itself.
So these feed it values that must be refused and copies that must not match.

The end-to-end test needs PostgreSQL and makes a database of its own, which
it drops afterwards; it never touches the server's `voicebot` database.
"""

from __future__ import annotations

import importlib.util
import random
import sys
import uuid
from datetime import datetime
from pathlib import Path
from types import ModuleType

import pytest

from tests import support

pytest.importorskip("psycopg")

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "migrate_sqlite_to_postgres.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("migrate_sqlite_to_postgres", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


script = _load()
TURNS = next(t for t in script.TABLES if t.name == "turns")


# --- values the source must not be allowed to hold ------------------------------


@pytest.mark.parametrize(
    ("convert", "value"),
    [
        (script._boolean, 2),
        (script._boolean, "1"),
        (script._boolean, 1.0),
        (script._integer, "5"),
        (script._integer, True),
        (script._integer, 2.5),
        (script._text, 5),
        (script._stamp, "2026/10/27 01:00:00"),
        (script._stamp, "2026-10-27"),
        (script._stamp, ""),
        (script._real, "fast"),
    ],
)
def test_a_value_that_does_not_fit_its_column_is_refused(convert, value) -> None:
    with pytest.raises(script.Refused):
        convert(value, "table.column")


def test_an_empty_lock_becomes_no_lock() -> None:
    assert script._stamp_or_empty("", "cards.pin_locked_until") is None


def test_a_flag_becomes_a_boolean_not_an_integer() -> None:
    assert script._boolean(1, "cards.frozen") is True
    assert script._boolean(0, "cards.frozen") is False


# --- copies that must not be called equal ---------------------------------------


def _row(**changes) -> tuple:
    base = {
        "id": 1,
        "session_id": "s1",
        "turn_index": 1,
        "user_text": "hello",
        "bot_text": "hi — there",
        "tool_calls": None,
        "guard_hit": False,
        "suppressed": None,
        "latency_ms": 12.5,
        "created_at": datetime(2026, 10, 28, 19, 33, 49),
    }
    base.update(changes)
    return tuple(base[c] for c in TURNS.target_columns)


def _compare(source: list[tuple], target: list[tuple]):
    return script.compare(TURNS, source, target, random.Random(0))


def test_identical_copies_match() -> None:
    assert _compare([_row()], [_row()]).ok


def test_one_changed_character_is_caught() -> None:
    result = _compare([_row()], [_row(bot_text="hi - there")])
    assert not result.ok
    assert any("bot_text" in m for m in result.mismatches)


def test_one_versus_true_is_caught_although_python_calls_them_equal() -> None:
    result = _compare([_row(guard_hit=True)], [_row(guard_hit=1)])
    assert not result.ok


def test_a_missing_row_is_caught() -> None:
    result = _compare([_row(), _row(id=2)], [_row()])
    assert not result.ok
    assert result.source_count != result.target_count


def test_a_null_that_became_an_empty_string_is_caught() -> None:
    assert not _compare([_row(tool_calls=None)], [_row(tool_calls="")]).ok


def test_the_checksum_ignores_row_order_and_nothing_else() -> None:
    a, b = _row(), _row(id=2)
    assert script.checksum([a, b]) == script.checksum([b, a])
    assert script.checksum([a]) != script.checksum([_row(latency_ms=12.500001)])


# --- the whole run, refusing a target that has drifted --------------------------


@pytest.mark.skipif(not support.TESTING_POSTGRES, reason="needs PostgreSQL")
def test_a_tampered_target_is_refused_until_replaced(tmp_path, monkeypatch) -> None:
    import psycopg
    from psycopg import sql

    from voicebot.bank.postgres_bank import conninfo_from_env
    from voicebot.bank.sqlite_bank import SqliteBank

    source = tmp_path / "bank.db"
    SqliteBank(str(source)).close()
    name = f"voicebot_migration_test_{uuid.uuid4().hex[:10]}"
    monkeypatch.setenv("VOICEBOT_DB_NAME", name)
    report = tmp_path / "report.md"
    try:
        assert script.main(["--source", str(source), "--report", str(report)]) == 0
        assert "PASS" in report.read_text(encoding="utf-8")
        assert script.main(["--source", str(source)]) == 0  # already up to date

        with psycopg.connect(conninfo_from_env(), autocommit=True) as conn:
            conn.execute("UPDATE accounts SET balance_cents = balance_cents + 1")

        assert script.main(["--source", str(source)]) == 1  # refused: it drifted
        assert script.main(["--source", str(source), "--replace"]) == 0
    finally:
        with psycopg.connect(conninfo_from_env(dbname="postgres"), autocommit=True) as conn:
            conn.execute(
                sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name))
            )
