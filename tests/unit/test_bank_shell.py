"""`python -m voicebot.bank`: the shell commands that look after the demo bank.

They must act on the bank the server opens. That used to be data/bank.db
whatever the configuration said; with the bank in PostgreSQL, a deposit made
into the file is money the assistant never sees.

Every test runs in its own directory, so neither the project's .env nor its
configs/serve.yaml is ever the one read.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from voicebot.bank import SqliteBank, open_bank
from voicebot.bank.base import Bank, StorageError
from voicebot.bank.seed import ACCOUNTS, main
from voicebot.config import DatabaseConfig

OPENING_CHECKING = dict(ACCOUNTS)["checking"]


@pytest.fixture(autouse=True)
def elsewhere(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    # Registered before it is removed, so that monkeypatch puts it back as it
    # was however a test's .env leaves it.
    monkeypatch.setenv("VOICEBOT_CONFIG", "unset")
    monkeypatch.delenv("VOICEBOT_CONFIG")
    return tmp_path


def _config(folder: Path, database: str, engine: str = "sqlite") -> Path:
    path = folder / "bank.yaml"
    path.write_text(f"database:\n  engine: {engine}\n  path: {database}\n", encoding="utf-8")
    return path


# --- open_bank ---------------------------------------------------------------


def test_open_bank_opens_the_sqlite_file_it_is_given(elsewhere: Path) -> None:
    bank = open_bank(DatabaseConfig(engine="sqlite", path="bank.db"))

    assert isinstance(bank, SqliteBank)
    assert (elsewhere / "bank.db").is_file()


def test_open_bank_reaches_postgres_through_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("psycopg")
    from voicebot.bank import postgres_bank

    opened: dict[str, object] = {}

    def recorder(conninfo: str, seed_if_empty: bool, pool_size: int) -> None:
        opened.update(conninfo=conninfo, seed_if_empty=seed_if_empty, pool_size=pool_size)

    monkeypatch.setattr(postgres_bank, "PostgresBank", recorder)
    monkeypatch.setenv("VOICEBOT_DB_NAME", "branch_office")

    open_bank(DatabaseConfig(engine="postgres", seed_on_create=False, pool_size=3))

    assert "dbname=branch_office" in str(opened["conninfo"])
    assert opened["seed_if_empty"] is False
    assert opened["pool_size"] == 3


def test_open_bank_refuses_an_engine_it_does_not_know() -> None:
    with pytest.raises(ValueError, match="unknown database engine 'postgress'"):
        open_bank(DatabaseConfig(engine="postgress"))


# --- which bank the shell acts on -------------------------------------------


def test_a_deposit_lands_in_the_bank_the_configuration_names(
    elsewhere: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["--config", str(_config(elsewhere, "live.db")), "--deposit", "5"])

    assert SqliteBank(path="live.db").balance("checking") == OPENING_CHECKING + 500
    assert "live.db" in capsys.readouterr().out


def test_with_no_flag_it_follows_VOICEBOT_CONFIG_like_the_server(
    elsewhere: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VOICEBOT_CONFIG", str(_config(elsewhere, "from-the-environment.db")))

    main(["--show"])

    assert (elsewhere / "from-the-environment.db").is_file()


def test_dot_env_counts_as_it_does_for_the_server(elsewhere: Path) -> None:
    _config(elsewhere, "from-dot-env.db")
    (elsewhere / ".env").write_text("VOICEBOT_CONFIG=bank.yaml\n", encoding="utf-8")

    main(["--show"])

    assert (elsewhere / "from-dot-env.db").is_file()


def test_path_names_a_sqlite_file_outright(
    elsewhere: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("VOICEBOT_CONFIG", str(_config(elsewhere, "configured.db")))

    main(["--path", "named.db", "--reset"])

    assert (elsewhere / "named.db").is_file()
    assert not (elsewhere / "configured.db").exists()
    out = capsys.readouterr().out
    assert "created named.db" in out
    assert "reset named.db to its opening balances" in out


class _StandIn(SqliteBank):
    """An in-memory bank that remembers being closed."""

    closed = False

    def close(self) -> None:
        self.closed = True


def test_on_postgres_it_names_the_database_but_never_the_password(
    elsewhere: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("psycopg")
    monkeypatch.setenv("VOICEBOT_DB_USER", "teller")
    monkeypatch.setenv("VOICEBOT_DB_NAME", "branch_office")
    monkeypatch.setenv("VOICEBOT_DB_PASSWORD", "hunter2-never-printed")
    asked_for: list[DatabaseConfig] = []
    stand_in = _StandIn()

    def open_postgres(database: DatabaseConfig) -> Bank:
        asked_for.append(database)
        return stand_in

    monkeypatch.setattr("voicebot.bank.open_bank", open_postgres)

    main(["--config", str(_config(elsewhere, "unused.db", engine="postgres")), "--show"])

    out = capsys.readouterr().out
    assert asked_for[0].engine == "postgres"
    assert "PostgreSQL teller@localhost:5432/branch_office" in out
    assert "hunter2" not in out
    assert stand_in.closed


def test_a_bank_it_cannot_open_is_a_message_not_a_traceback(
    elsewhere: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def refuse(database: DatabaseConfig) -> Bank:
        raise StorageError("could not connect to PostgreSQL: password authentication failed")

    monkeypatch.setattr("voicebot.bank.open_bank", refuse)

    with pytest.raises(SystemExit) as stopped:
        main(["--config", str(_config(elsewhere, "unused.db")), "--show"])

    assert stopped.value.code == 1
    assert "password authentication failed" in capsys.readouterr().err
