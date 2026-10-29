"""The turn record: written after the turn, retried, and never silently lost."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from voicebot.api.audit import AuditWriter, TurnRecord
from voicebot.bank.base import StorageError

RECORD = TurnRecord("s1", 1, "what is my balance", "$19,020.00.", None, False, None, 900.0)


class Flaky:
    """A bank whose log_turn fails `times` times, then works."""

    def __init__(self, times: int) -> None:
        self.times = times
        self.written: list[dict[str, Any]] = []

    def log_turn(self, **row: Any) -> None:
        if self.times:
            self.times -= 1
            raise StorageError("connection lost")
        self.written.append(row)


def test_a_record_is_retried_until_the_database_takes_it(tmp_path: Path) -> None:
    bank = Flaky(times=2)
    writer = AuditWriter(bank, tmp_path / "dead.jsonl", retries=3, sleep=lambda _: None)  # type: ignore[arg-type]
    writer.submit(RECORD)
    assert writer.flush()
    assert len(bank.written) == 1 and writer.dead == 0
    assert not (tmp_path / "dead.jsonl").exists()
    writer.close()


def test_a_full_queue_keeps_the_record_without_making_the_caller_wait(tmp_path: Path) -> None:
    release = threading.Event()

    class Stalled:
        def log_turn(self, **row: Any) -> None:
            release.wait(5)

    dead = tmp_path / "dead.jsonl"
    writer = AuditWriter(Stalled(), dead, queue_size=1, sleep=lambda _: None)  # type: ignore[arg-type]
    for _ in range(4):
        writer.submit(RECORD)  # returns at once, every time
    assert writer.dead >= 2
    assert all(json.loads(line)["why"] for line in dead.read_text(encoding="utf-8").splitlines())
    release.set()
    writer.close()


def test_records_still_queued_at_shutdown_are_kept(tmp_path: Path) -> None:
    release = threading.Event()

    class Slow:
        def log_turn(self, **row: Any) -> None:
            release.wait(2)

    dead = tmp_path / "dead.jsonl"
    writer = AuditWriter(Slow(), dead, sleep=lambda _: None)  # type: ignore[arg-type]
    for _ in range(3):
        writer.submit(RECORD)
    writer.close(timeout_s=0.1)
    release.set()
    assert writer.dead >= 1 and dead.exists()


def test_when_even_the_file_fails_the_loss_is_logged_without_the_words(
    tmp_path: Path, caplog: Any
) -> None:
    records: list[logging.LogRecord] = []

    class Keep(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = Keep()
    logging.getLogger("voicebot").addHandler(handler)
    try:
        blocked = tmp_path / "a-file-not-a-folder"
        blocked.write_text("x")
        bank = Flaky(times=99)
        writer = AuditWriter(bank, blocked / "dead.jsonl", retries=0, sleep=lambda _: None)  # type: ignore[arg-type]
        writer.submit(RECORD)
        assert writer.flush()
        writer.close()
    finally:
        logging.getLogger("voicebot").removeHandler(handler)
    lost = [r.getMessage() for r in records if "was lost" in r.getMessage()]
    assert lost and "balance" not in lost[0]
