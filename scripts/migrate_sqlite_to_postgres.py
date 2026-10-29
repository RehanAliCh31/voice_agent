"""Copy the SQLite bank into PostgreSQL, and prove the copy is exact.

    python scripts/migrate_sqlite_to_postgres.py                  # data/bank.db -> VOICEBOT_DB_NAME
    python scripts/migrate_sqlite_to_postgres.py --dry-run        # do it all, then roll back
    python scripts/migrate_sqlite_to_postgres.py --replace        # overwrite a target that differs
    python scripts/migrate_sqlite_to_postgres.py --source data/backups/bank-20261029-103529.db

Connection settings come from VOICEBOT_DB_* in the environment or in .env,
exactly as the server reads them. Nothing here prints the password.

What it guarantees:

* **The SQLite file is only ever read.** It is opened with `mode=ro`, and
  its SHA-256 is taken before and after; a difference stops the run.
* **Nothing is guessed.** Every column of every table is declared below with
  the type it must hold. A column the file has that is not declared, or a
  value that does not fit its declaration, stops the run before anything is
  written.
* **All or nothing.** The whole copy is one transaction, and the
  verification runs inside it: counts, checksums and a field-by-field
  comparison of every row. Any mismatch rolls the transaction back, so the
  target is never left half-migrated.
* **Safe to run again.** A target that already holds exactly this data is
  reported as up to date and left alone. A target holding anything else is
  refused unless --replace is given.
* **A dry run keeps nothing.** It copies and verifies everything, then rolls
  back -- and puts the identity sequences back too, because `setval` is not
  undone by a rollback. The one thing it does keep is the empty database and
  its schema, if it had to create them: both are what a real run needs.

Exit codes: 0 migrated or already up to date, 1 refused or failed
verification, 2 could not connect or bad arguments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sqlite3
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import tuple_row

from voicebot.bank.migrations import migrate
from voicebot.bank.postgres_bank import conninfo_from_env
from voicebot.utils.env import load_env_file

ROOT = Path(__file__).resolve().parent.parent
BATCH = 500
STAMP = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")


class Refused(Exception):
    """Something about the source or target is not what this script expects."""


# --- what each column must hold ------------------------------------------------
#
# A converter takes the SQLite value and returns the PostgreSQL one, or raises
# Refused. The same converter turns a SQLite row into the form it is compared
# in, so the comparison is between like and like.


def _text(value: Any, column: str) -> Any:
    if value is not None and not isinstance(value, str):
        raise Refused(f"{column}: expected text, found {type(value).__name__} {value!r}")
    return value


def _integer(value: Any, column: str) -> Any:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
        raise Refused(f"{column}: expected an integer, found {type(value).__name__} {value!r}")
    return value


def _boolean(value: Any, column: str) -> bool:
    if value not in (0, 1) or isinstance(value, float):
        raise Refused(f"{column}: expected 0 or 1 for a boolean, found {value!r}")
    return bool(value)


def _real(value: Any, column: str) -> Any:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise Refused(f"{column}: expected a number, found {type(value).__name__} {value!r}")
    return float(value)


def _stamp(value: Any, column: str) -> datetime:
    if not isinstance(value, str) or not STAMP.match(value):
        raise Refused(f"{column}: expected 'YYYY-MM-DD HH:MM:SS', found {value!r}")
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def _stamp_or_empty(value: Any, column: str) -> datetime | None:
    """SQLite's '' for "no value", which a timestamp column holds as NULL."""
    return None if value == "" else _stamp(value, column)


Converter = Callable[[Any, str], Any]


@dataclass(frozen=True)
class Table:
    name: str
    key: tuple[str, ...]
    columns: dict[str, Converter]
    # Identity columns whose sequences are moved past the copied values.
    identity: tuple[str, ...] = ()
    # SQLite's hidden rowid, copied into this column where the code orders by it.
    rowid_as: str | None = None

    @property
    def target_columns(self) -> list[str]:
        return list(self.columns) + ([self.rowid_as] if self.rowid_as else [])


# Foreign-key order: every table after the tables it refers to.
TABLES: tuple[Table, ...] = (
    Table("accounts", ("name",), {"name": _text, "balance_cents": _integer}),
    Table("payees", ("name",), {"name": _text}),
    Table(
        "bills",
        ("name",),
        {
            "name": _text,
            "amount_cents": _integer,
            "paid": _boolean,
            "category": _text,
            "due_day": _integer,
        },
    ),
    Table(
        "cards",
        ("last4",),
        {
            "last4": _text,
            "account": _text,
            "frozen": _boolean,
            "kind": _text,
            "holder": _text,
            "number": _text,
            "valid_from": _text,
            "valid_to": _text,
            "pin_hash": _text,
            "pin_salt": _text,
            "pin_failures": _integer,
            "pin_locked_until": _stamp_or_empty,
            "ecommerce": _boolean,
            "retail": _boolean,
            "atm": _boolean,
        },
    ),
    Table(
        "transactions",
        ("id",),
        {
            "id": _integer,
            "account": _text,
            "merchant": _text,
            "cents": _integer,
            "occurred_at": _stamp,
        },
        identity=("id",),
    ),
    Table(
        "disputes",
        ("reference",),
        {
            "reference": _text,
            "transaction_id": _integer,
            "merchant": _text,
            "amount_cents": _integer,
            "occurred_at": _stamp,
            "reason": _text,
            "note": _text,
            "status": _text,
            "created_at": _stamp,
        },
        identity=("seq",),
        rowid_as="seq",
    ),
    Table(
        "handoffs",
        ("reference",),
        {
            "reference": _text,
            "session_id": _text,
            "turn_index": _integer,
            "reason": _text,
            "note": _text,
            "created_at": _stamp,
        },
        identity=("seq",),
        rowid_as="seq",
    ),
    Table(
        "turns",
        ("id",),
        {
            "id": _integer,
            "session_id": _text,
            "turn_index": _integer,
            "user_text": _text,
            "bot_text": _text,
            "tool_calls": _text,
            "guard_hit": _boolean,
            "suppressed": _text,
            "latency_ms": _real,
            "created_at": _stamp,
        },
        identity=("id",),
    ),
)

# Never printed in the report: a card number and the PIN's hash and salt.
SENSITIVE = {"number", "pin_hash", "pin_salt"}
# Free text a customer said or was told; the report shows its length only.
CONVERSATION = {"user_text", "bot_text", "suppressed", "note"}


def scalar(cursor: Any) -> Any:
    """The one value a statement is known to return. A missing row is a fault."""
    row = cursor.fetchone()
    if row is None:
        raise Refused("the database returned no row where one is always expected")
    return row[0]


# --- reading the source --------------------------------------------------------


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def open_source(path: Path) -> sqlite3.Connection:
    """Read-only, by URI: this connection cannot write the file."""
    if not path.is_file():
        raise Refused(f"no SQLite file at {path}")
    db = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    if (result := scalar(db.execute("PRAGMA integrity_check"))) != "ok":
        raise Refused(f"{path} fails SQLite's integrity check: {result}")
    if violations := db.execute("PRAGMA foreign_key_check").fetchall():
        raise Refused(f"{path} has foreign-key violations: {violations[:5]}")
    return db


def source_rows(db: sqlite3.Connection, table: Table) -> list[tuple[Any, ...]]:
    """Every row of one table, converted, in target column order."""
    present = [row[1] for row in db.execute(f"PRAGMA table_info({table.name})")]
    unexpected = sorted(set(present) - set(table.columns))
    missing = sorted(set(table.columns) - set(present))
    if unexpected or missing:
        raise Refused(
            f"{table.name}: columns differ from what this script knows "
            f"(unexpected {unexpected or 'none'}, missing {missing or 'none'})"
        )
    names = list(table.columns)
    select = ", ".join(names + (["rowid"] if table.rowid_as else []))
    rows = []
    for raw in db.execute(f"SELECT {select} FROM {table.name}"):
        declared = raw[: len(names)]
        converted = [
            convert(value, f"{table.name}.{column}")
            for (column, convert), value in zip(table.columns.items(), declared, strict=True)
        ]
        if table.rowid_as:
            converted.append(raw[-1])
        rows.append(tuple(converted))
    return rows


def sqlite_counter(db: sqlite3.Connection, table: str) -> int | None:
    """What SQLite's AUTOINCREMENT would hand out next, minus one."""
    row = db.execute("SELECT seq FROM sqlite_sequence WHERE name = ?", (table,)).fetchone()
    return None if row is None else int(row[0])


# --- the target ----------------------------------------------------------------


def describe(conninfo: str) -> str:
    """host:port/dbname as user -- everything but the password."""
    parts = conninfo_to_dict(conninfo)
    return (
        f"{parts.get('user', '(default user)')}@{parts.get('host', 'localhost')}:"
        f"{parts.get('port', '5432')}/{parts.get('dbname', '?')}"
    )


def ensure_database(conninfo: str, log: list[str]) -> None:
    """Create the target database if it is missing, as UTF-8 with C collation.

    UTF-8 because the source holds non-ASCII text; C collation because it
    orders text byte for byte, exactly as SQLite does.
    """
    name = str(conninfo_to_dict(conninfo)["dbname"])
    admin = conninfo_from_env(dbname="postgres")
    with psycopg.connect(admin, autocommit=True) as conn:
        found = conn.execute(
            "SELECT pg_encoding_to_char(encoding), datcollate FROM pg_database WHERE datname = %s",
            (name,),
        ).fetchone()
        if found is None:
            conn.execute(
                sql.SQL(
                    "CREATE DATABASE {} WITH ENCODING 'UTF8' "
                    "LC_COLLATE 'C' LC_CTYPE 'C' TEMPLATE template0"
                ).format(sql.Identifier(name))
            )
            log.append(f"created database {name} (UTF8, C collation)")
            return
        encoding, collation = found
        if encoding != "UTF8":
            raise Refused(
                f"database {name} is {encoding}, not UTF8. The source holds non-ASCII text "
                "that would not survive; create the database as UTF8 and run again."
            )
        log.append(f"database {name} exists ({encoding}, collation {collation})")
        if collation != "C":
            log.append(
                f"  note: collation {collation} may order text differently from SQLite; "
                "C would match it exactly"
            )


def target_rows(cur: psycopg.Cursor[Any], table: Table) -> list[tuple[Any, ...]]:
    query = sql.SQL("SELECT {} FROM {}").format(
        sql.SQL(", ").join(map(sql.Identifier, table.target_columns)),
        sql.Identifier(table.name),
    )
    return [tuple(row) for row in cur.execute(query)]


def copy_table(cur: psycopg.Cursor[Any], table: Table, rows: Sequence[tuple[Any, ...]]) -> None:
    insert = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
        sql.Identifier(table.name),
        sql.SQL(", ").join(map(sql.Identifier, table.target_columns)),
        sql.SQL(", ").join(sql.Placeholder() * len(table.target_columns)),
    )
    for start in range(0, len(rows), BATCH):
        cur.executemany(insert, rows[start : start + BATCH])


def move_sequences(cur: psycopg.Cursor[Any], table: Table, floor: int | None) -> list[str]:
    """Point each identity sequence past the highest copied value.

    `floor` is SQLite's own counter, which can be higher than the highest id
    still present if rows were deleted; AUTOINCREMENT never reuses one, and
    neither should the copy.
    """
    moved = []
    for column in table.identity:
        highest = scalar(
            cur.execute(
                sql.SQL("SELECT MAX({}) FROM {}").format(
                    sql.Identifier(column), sql.Identifier(table.name)
                )
            )
        )
        # `seq` stands in for rowid, which has no counter of its own.
        last = max(highest or 0, (floor or 0) if column != "seq" else 0)
        if last:
            cur.execute(
                "SELECT setval(pg_get_serial_sequence(%s, %s), %s, true)",
                (table.name, column, last),
            )
        else:
            cur.execute(
                "SELECT setval(pg_get_serial_sequence(%s, %s), 1, false)", (table.name, column)
            )
        moved.append(f"{table.name}.{column} next value {last + 1}")
    return moved


def save_sequences(cur: psycopg.Cursor[Any]) -> dict[str, tuple[int, bool]]:
    """Where every identity sequence stands, so it can be put back.

    Sequences are not transactional: `setval` survives a rollback. A dry run
    or a failed verification rolls the rows back, and without this would
    leave the sequences moved -- the next real insert would skip numbers.
    """
    saved = {}
    for table in TABLES:
        for column in table.identity:
            name = scalar(
                cur.execute("SELECT pg_get_serial_sequence(%s, %s)", (table.name, column))
            )
            schema, _, relation = name.rpartition(".")
            state = cur.execute(
                sql.SQL("SELECT last_value, is_called FROM {}").format(
                    sql.Identifier(schema, relation) if schema else sql.Identifier(relation)
                )
            ).fetchone()
            if state is None:
                raise Refused(f"could not read where sequence {name} stands")
            saved[name] = (int(state[0]), bool(state[1]))
    return saved


def restore_sequences(cur: psycopg.Cursor[Any], saved: dict[str, tuple[int, bool]]) -> None:
    for name, (last_value, is_called) in saved.items():
        cur.execute("SELECT setval(%s, %s, %s)", (name, last_value, is_called))


# --- proving it ----------------------------------------------------------------


def canonical(value: Any) -> Any:
    """One form for hashing, the same whichever side the value came from."""
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, float):
        return repr(value)
    return value


def checksum(rows: Iterable[tuple[Any, ...]]) -> str:
    """SHA-256 over every row, sorted, in a form independent of the driver."""
    lines = sorted(json.dumps([canonical(v) for v in row], ensure_ascii=False) for row in rows)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


@dataclass
class TableResult:
    name: str
    source_count: int
    target_count: int
    source_checksum: str
    target_checksum: str
    mismatches: list[str] = field(default_factory=list)
    aggregates: list[tuple[str, Any, Any]] = field(default_factory=list)
    samples: list[tuple[str, list[tuple[str, Any, Any, bool]]]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (
            self.source_count == self.target_count
            and self.source_checksum == self.target_checksum
            and not self.mismatches
            and all(a == b for _, a, b in self.aggregates)
        )


def compare(
    table: Table,
    source: list[tuple[Any, ...]],
    target: list[tuple[Any, ...]],
    rng: random.Random,
) -> TableResult:
    columns = table.target_columns
    key_at = [columns.index(k) for k in table.key]
    by_key_s = {tuple(row[i] for i in key_at): row for row in source}
    by_key_t = {tuple(row[i] for i in key_at): row for row in target}

    result = TableResult(
        name=table.name,
        source_count=len(source),
        target_count=len(target),
        source_checksum=checksum(source),
        target_checksum=checksum(target),
    )

    # Every row, every field.
    for key in sorted(set(by_key_s) | set(by_key_t), key=repr):
        s, t = by_key_s.get(key), by_key_t.get(key)
        if s is None or t is None:
            result.mismatches.append(f"{key}: only in {'target' if s is None else 'source'}")
            continue
        for column, a, b in zip(columns, s, t, strict=True):
            if a != b or type(a) is not type(b):
                result.mismatches.append(f"{key}.{column}: source {a!r} != target {b!r}")

    # Aggregates, computed independently on each side.
    for i, column in enumerate(columns):
        values_s = [row[i] for row in source if row[i] is not None]
        values_t = [row[i] for row in target if row[i] is not None]
        result.aggregates.append((f"{column} non-null", len(values_s), len(values_t)))
        sample = values_s[0] if values_s else values_t[0] if values_t else None
        if isinstance(sample, int) and not isinstance(sample, bool):
            result.aggregates.append((f"SUM({column})", sum(values_s), sum(values_t)))
        elif isinstance(sample, datetime):
            result.aggregates.append(
                (
                    f"MIN/MAX({column})",
                    (canonical(min(values_s)), canonical(max(values_s))) if values_s else None,
                    (canonical(min(values_t)), canonical(max(values_t))) if values_t else None,
                )
            )
        elif isinstance(sample, bool):
            result.aggregates.append((f"COUNT({column} = true)", sum(values_s), sum(values_t)))

    # A few rows shown side by side, for a human to read.
    for key in rng.sample(sorted(by_key_s, key=repr), k=min(3, len(by_key_s))):
        s, t = by_key_s[key], by_key_t.get(key)
        fields = []
        for i, column in enumerate(columns):
            a, b = s[i], (t[i] if t else None)
            fields.append((column, a, b, t is not None and a == b))
        result.samples.append((repr(key), fields))
    return result


def orphans(cur: psycopg.Cursor[Any]) -> list[tuple[str, int]]:
    checks = {
        "cards.account with no account": "SELECT COUNT(*) FROM cards c "
        "LEFT JOIN accounts a ON a.name = c.account WHERE a.name IS NULL",
        "transactions.account with no account": "SELECT COUNT(*) FROM transactions t "
        "LEFT JOIN accounts a ON a.name = t.account WHERE a.name IS NULL",
        "disputes.transaction_id with no transaction (no FK; reported only)": "SELECT COUNT(*) "
        "FROM disputes d LEFT JOIN transactions t ON t.id = d.transaction_id WHERE t.id IS NULL",
    }
    return [(label, scalar(cur.execute(query))) for label, query in checks.items()]


def invalid_constraints(cur: psycopg.Cursor[Any]) -> list[str]:
    """Any constraint PostgreSQL has not validated -- there should be none."""
    return [
        f"{row[0]}.{row[1]}"
        for row in cur.execute(
            "SELECT conrelid::regclass::text, conname FROM pg_constraint "
            "WHERE connamespace = current_schema()::regnamespace AND NOT convalidated"
        )
    ]


# --- the run -------------------------------------------------------------------


def verify(
    db: sqlite3.Connection,
    cur: psycopg.Cursor[Any],
    source_data: dict[str, list[tuple[Any, ...]]],
    seed: int = 20261029,
) -> list[TableResult]:
    rng = random.Random(seed)
    return [compare(t, source_data[t.name], target_rows(cur, t), rng) for t in TABLES]


def run(args: argparse.Namespace) -> int:
    started = datetime.now(UTC)
    try:
        return _run(args, started)
    except _VerificationFailed as exc:
        source = Path(args.source) if Path(args.source).is_absolute() else ROOT / args.source
        digest = sha256(source)
        report(
            args,
            exc.log,
            exc.results,
            exc.found_orphans,
            exc.unvalidated,
            started,
            digest,
            digest,
            status="FAILED, rolled back",
        )
        raise


def _run(args: argparse.Namespace, started: datetime) -> int:
    load_env_file(ROOT / ".env")
    source = Path(args.source)
    if not source.is_absolute():
        source = ROOT / source
    log: list[str] = []

    before = sha256(source)
    db = open_source(source)
    log.append(f"source {source} (sha256 {before[:16]}...), opened read-only")

    # Read and convert everything before touching the target, so a value
    # that does not fit stops the run with nothing written.
    source_data = {t.name: source_rows(db, t) for t in TABLES}
    counters = {t.name: sqlite_counter(db, t.name) for t in TABLES}
    log.append(
        "source read and type-checked: "
        + ", ".join(f"{name} {len(rows)}" for name, rows in source_data.items())
    )

    conninfo = conninfo_from_env()
    log.append(f"target {describe(conninfo)}")
    ensure_database(conninfo, log)

    with psycopg.connect(conninfo, autocommit=True) as conn:
        server = scalar(conn.execute("SHOW server_version"))
        applied = migrate(conn)
        log.append(
            f"server PostgreSQL {server}; schema migrations applied now: {applied or 'none'}"
        )
        cur = conn.cursor(row_factory=tuple_row)

        present = {
            t.name: scalar(
                cur.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(t.name)))
            )
            for t in TABLES
        }
        if any(present.values()):
            existing = verify(db, cur, source_data)
            if all(r.ok for r in existing):
                log.append("target already holds exactly this data: nothing to do")
                results = existing
                report(
                    args,
                    log,
                    results,
                    orphans(cur),
                    invalid_constraints(cur),
                    started,
                    before,
                    sha256(source),
                    status="up to date",
                )
                return 0
            if not args.replace:
                differing = [r.name for r in existing if not r.ok]
                raise Refused(
                    f"the target already holds different data (in {', '.join(differing)}). "
                    "Run again with --replace to delete it and copy the source over it."
                )

        saved = save_sequences(cur)
        committed = False
        try:
            results, found_orphans, unvalidated, committed = _copy(
                conn, cur, db, args, present, source_data, counters, log
            )
        finally:
            if not committed:
                restore_sequences(cur, saved)
                log.append("sequences put back where they were: nothing kept")
        status = "dry run" if args.dry_run else "migrated"

    # A fresh connection, so what is checked is what was committed.
    if not args.dry_run:
        with psycopg.connect(conninfo, autocommit=True) as conn:
            cur = conn.cursor(row_factory=tuple_row)
            results = verify(db, cur, source_data)
            found_orphans = orphans(cur)
            unvalidated = invalid_constraints(cur)
            log.append("re-verified from a new connection after commit")

    after = sha256(source)
    db.close()
    if after != before:
        raise Refused(f"the source file changed during the run ({before[:16]} -> {after[:16]})")
    report(args, log, results, found_orphans, unvalidated, started, before, after, status=status)
    return 0 if all(r.ok for r in results) else 1


def _copy(
    conn: psycopg.Connection[Any],
    cur: psycopg.Cursor[Any],
    db: sqlite3.Connection,
    args: argparse.Namespace,
    present: dict[str, int],
    source_data: dict[str, list[tuple[Any, ...]]],
    counters: dict[str, int | None],
    log: list[str],
) -> tuple[list[TableResult], list[tuple[str, int]], list[str], bool]:
    """The copy, in one transaction, verified before it can commit.

    Returns the verification and whether it was committed -- False for a dry
    run, which rolls back on purpose. A failed verification raises, and the
    raise is what rolls the transaction back.
    """
    with conn.transaction() as tx:
        if any(present.values()):
            for table in reversed(TABLES):
                cur.execute(sql.SQL("DELETE FROM {}").format(sql.Identifier(table.name)))
            log.append("--replace: emptied the target tables inside the transaction")
        for table in TABLES:
            copy_table(cur, table, source_data[table.name])
            log.append(f"copied {table.name}: {len(source_data[table.name])} rows")
        for table in TABLES:
            for line in move_sequences(cur, table, counters[table.name]):
                log.append(f"sequence {line}")

        results = verify(db, cur, source_data)
        found_orphans = orphans(cur)
        unvalidated = invalid_constraints(cur)
        hard_orphans = [
            (label, n) for label, n in found_orphans if n and "reported only" not in label
        ]
        if not all(r.ok for r in results) or hard_orphans or unvalidated:
            log.append("VERIFICATION FAILED inside the transaction: rolled back, nothing written")
            raise _VerificationFailed(results, found_orphans, unvalidated, log)
        if args.dry_run:
            log.append("--dry-run: everything copied and verified, then rolled back on purpose")
            raise psycopg.Rollback(tx)
    return results, found_orphans, unvalidated, not args.dry_run


class _VerificationFailed(Exception):
    """Raised inside the transaction, which is what rolls it back."""

    def __init__(
        self,
        results: list[TableResult],
        found_orphans: list[tuple[str, int]],
        unvalidated: list[str],
        log: list[str],
    ) -> None:
        super().__init__("verification failed")
        self.results = results
        self.found_orphans = found_orphans
        self.unvalidated = unvalidated
        self.log = log


# --- the report ----------------------------------------------------------------


def shown(column: str, value: Any) -> str:
    if value is None:
        return "NULL"
    if column in SENSITIVE:
        return "(withheld)"
    if column in CONVERSATION and isinstance(value, str):
        return f"({len(value)} chars)"
    return repr(canonical(value)) if not isinstance(value, str) else repr(value)


def report(
    args: argparse.Namespace,
    log: list[str],
    results: list[TableResult],
    found_orphans: list[tuple[str, int]],
    unvalidated: list[str],
    started: datetime,
    before: str,
    after: str,
    status: str,
) -> None:
    passed = all(r.ok for r in results)
    lines = [
        "# SQLite to PostgreSQL: verification report",
        "",
        f"- **Run:** {started:%Y-%m-%d %H:%M:%S} UTC ({status})",
        f"- **Result:** {'PASS -- every table matches' if passed else 'FAIL'}",
        f"- **Source SHA-256 before:** `{before}`",
        f"- **Source SHA-256 after:**  `{after}` ({'unchanged' if before == after else 'CHANGED'})",
        "",
        "## Row counts and checksums",
        "",
        "| Table | SQLite rows | PostgreSQL rows | Rows checked field by field "
        "| Checksum | Status |",
        "|---|---:|---:|---:|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.name} | {r.source_count} | {r.target_count} | {r.source_count} | "
            f"{'match' if r.source_checksum == r.target_checksum else 'DIFFERENT'} "
            f"`{r.target_checksum[:12]}` | {'OK' if r.ok else 'MISMATCH'} |"
        )
    total_s = sum(r.source_count for r in results)
    total_t = sum(r.target_count for r in results)
    lines += [f"| **total** | **{total_s}** | **{total_t}** | **{total_s}** | | |", ""]

    lines += ["## Aggregates, computed independently on each side", ""]
    lines += ["| Table | Measure | SQLite | PostgreSQL | |", "|---|---|---|---|---|"]
    for r in results:
        for label, a, b in r.aggregates:
            if label.endswith("non-null") and a == r.source_count and b == r.target_count:
                continue
            lines.append(f"| {r.name} | {label} | {a} | {b} | {'=' if a == b else 'DIFFERENT'} |")
    lines.append("")

    lines += ["## Integrity", ""]
    for label, n in found_orphans:
        lines.append(f"- {label}: **{n}**")
    lines.append(
        f"- constraints not validated by PostgreSQL: **{len(unvalidated)}**"
        + (f" {unvalidated}" if unvalidated else "")
    )
    lines.append("")

    mismatches = [m for r in results for m in r.mismatches]
    lines += ["## Field-by-field differences", ""]
    lines.append("None." if not mismatches else "\n".join(f"- {m}" for m in mismatches[:200]))
    lines.append("")

    lines += [
        "## Sample rows, side by side",
        "",
        "Three per table, picked at random with a fixed seed. Card numbers and PIN hashes "
        "are withheld; conversation text is shown by length.",
        "",
    ]
    for r in results:
        for key, fields in r.samples:
            lines.append(f"**{r.name}** `{key}`")
            lines.append("")
            lines.append("| Column | SQLite | PostgreSQL | |")
            lines.append("|---|---|---|---|")
            for column, a, b, same in fields:
                verdict = "=" if same else "DIFFERENT"
                lines.append(f"| {column} | {shown(column, a)} | {shown(column, b)} | {verdict} |")
            lines.append("")

    lines += ["## Log", "", "```", *log, "```", ""]
    text = "\n".join(lines)
    if args.report:
        path = Path(args.report)
        if not path.is_absolute():
            path = ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"report written to {path}")

    print("\n".join(log))
    print()
    print(f"{'table':<14}{'sqlite':>8}{'postgres':>10}  status")
    for r in results:
        print(
            f"{r.name:<14}{r.source_count:>8}{r.target_count:>10}  {'OK' if r.ok else 'MISMATCH'}"
        )
    print(f"{'total':<14}{total_s:>8}{total_t:>10}  {'PASS' if passed else 'FAIL'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", default="data/bank.db", help="the SQLite file to copy")
    parser.add_argument("--replace", action="store_true", help="overwrite a target that differs")
    parser.add_argument(
        "--dry-run", action="store_true", help="copy and verify, then roll everything back"
    )
    parser.add_argument("--report", help="write the verification report to this Markdown file")
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    except _VerificationFailed as exc:
        print(
            "VERIFICATION FAILED: the transaction was rolled back and nothing was written.",
            file=sys.stderr,
        )
        for r in exc.results:
            for m in r.mismatches[:20]:
                print(f"  {r.name}: {m}", file=sys.stderr)
        return 1
    except psycopg.OperationalError as exc:
        print(f"could not reach PostgreSQL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
