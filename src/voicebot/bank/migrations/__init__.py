"""The PostgreSQL bank's schema, one numbered file per version.

Each `NNNN_description.sql` here is applied once, in order, and recorded in
`schema_migrations`. A schema change is a new file with the next number --
never an edit to one that has already run somewhere, because a database that
recorded it as applied will not run it again.

All pending files run in one transaction. PostgreSQL's DDL is transactional,
so a failure part way through leaves the schema exactly as it was rather
than half-changed. An advisory lock makes two servers starting at once take
turns instead of both trying to create the same tables.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from voicebot.utils.logging import get_logger

if TYPE_CHECKING:
    import psycopg

log = get_logger("bank.migrations")

DIRECTORY = Path(__file__).parent

# Any constant will do, as long as nothing else on the server uses it. Spells
# "voicebot" in ASCII, which makes it recognisable in pg_locks.
_LOCK_KEY = 0x766F696365626F74


def versions() -> list[Path]:
    """Every migration file, in the order it applies."""
    return sorted(DIRECTORY.glob("[0-9][0-9][0-9][0-9]_*.sql"))


def migrate(conn: psycopg.Connection) -> list[str]:
    """Apply whatever has not been applied yet. Returns what was applied."""
    from psycopg.rows import tuple_row

    applied_now: list[str] = []
    with conn.transaction():
        # Its own cursor, so this reads plain tuples whatever row factory the
        # caller's connection was opened with.
        cur = conn.cursor(row_factory=tuple_row)
        cur.execute("SELECT pg_advisory_xact_lock(%s)", (_LOCK_KEY,))
        cur.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            " version    TEXT PRIMARY KEY,"
            " applied_at TIMESTAMP(0) NOT NULL"
            "  DEFAULT date_trunc('second', now() AT TIME ZONE 'UTC'))"
        )
        done = {version for (version,) in cur.execute("SELECT version FROM schema_migrations")}
        for path in versions():
            if path.stem in done:
                continue
            log.info("applying schema migration %s", path.stem)
            # No parameters, so psycopg sends the file as one simple query and
            # PostgreSQL runs every statement in it.
            cur.execute(path.read_text(encoding="utf-8"))
            cur.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (path.stem,))
            applied_now.append(path.stem)
    return applied_now
