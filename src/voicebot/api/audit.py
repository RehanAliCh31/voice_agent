"""The record of each turn, written after the turn and never in its way.

A turn has done its work -- money moved, a case opened -- before its record is
written. Writing that record inside the request meant a database that failed
at that moment turned a finished turn into a 500, the customer saw an error,
and tried again: a second transfer. So the record goes on a queue, and a
worker writes it, retrying with backoff while the database recovers.

A record the database still will not take is not dropped. It is appended to
a dead-letter file, one JSON line each, counted, and alerted on, so it can be
replayed into the turns table once the database is back. Nothing about a
failed save is silent.
"""

from __future__ import annotations

import json
import queue
import random
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from voicebot.bank.base import Bank
from voicebot.errors import DatabaseError, call_context
from voicebot.utils.logging import get_logger
from voicebot.utils.observability import count, report, send_alert

log = get_logger("api.audit")


@dataclass(frozen=True)
class TurnRecord:
    """One row of the turns table, as Bank.log_turn takes it."""

    session_id: str
    turn_index: int
    user_text: str
    bot_text: str
    tool_calls: str | None
    guard_hit: bool
    suppressed: str | None
    latency_ms: float


class AuditWriter:
    """Writes turn records in the background, with retries and a dead-letter file."""

    def __init__(
        self,
        bank: Bank,
        dead_letter: str | Path,
        retries: int = 3,
        queue_size: int = 1000,
        backoff_s: float = 0.5,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.bank = bank
        self.dead_letter = Path(dead_letter)
        self.retries = retries
        self.backoff_s = backoff_s
        self._sleep = sleep or time.sleep
        self._queue: queue.Queue[TurnRecord | None] = queue.Queue(maxsize=queue_size)
        self._lock = threading.Lock()
        self.written = 0
        self.dead = 0
        self._worker = threading.Thread(target=self._run, name="audit-writer", daemon=True)
        self._worker.start()

    def submit(self, record: TurnRecord) -> None:
        """Queue `record` for writing. Never blocks and never raises."""
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            # The database has been down for a long while. The record is kept,
            # but not by waiting for it: the caller is still on the line.
            self._dead_letter(record, "the write queue was full")

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    def flush(self, timeout_s: float = 5.0) -> bool:
        """Wait until everything queued has been written or kept. True if it was."""
        finish = time.monotonic() + timeout_s
        while self._queue.unfinished_tasks and time.monotonic() < finish:
            time.sleep(0.005)
        return not self._queue.unfinished_tasks

    def close(self, timeout_s: float = 5.0) -> None:
        """Write what is queued, then stop. Called once, at shutdown."""
        self._queue.put(None)
        self._worker.join(timeout_s)
        # Whatever is still waiting when time runs out goes to the file.
        while True:
            try:
                record = self._queue.get_nowait()
            except queue.Empty:
                break
            if record is not None:
                self._dead_letter(record, "the service stopped before it was written")

    # --- the worker ----------------------------------------------------------

    def _run(self) -> None:
        while True:
            record = self._queue.get()
            try:
                if record is None:
                    return
                # Every line the worker logs belongs to the call the record is from.
                with call_context(record.session_id):
                    self._write(record)
            finally:
                self._queue.task_done()

    def _write(self, record: TurnRecord) -> None:
        for attempt in range(self.retries + 1):
            try:
                self.bank.log_turn(**asdict(record))
            except Exception as exc:  # any failure: retried, then kept -- never lost
                error = DatabaseError(
                    f"the turn record could not be written: {type(exc).__name__}",
                    component="audit",
                    cause=exc,
                )
                if attempt < self.retries:
                    report(error, attempt=attempt + 1, note="retrying")
                    self._sleep(random.uniform(0, self.backoff_s * 2**attempt))
                    continue
                report(error, attempt=attempt + 1)
                self._dead_letter(record, f"{type(exc).__name__} after {attempt + 1} attempts")
                return
            with self._lock:
                self.written += 1
            return

    def _dead_letter(self, record: TurnRecord, why: str) -> None:
        line = json.dumps({**asdict(record), "why": why, "at": time.time()}, ensure_ascii=False)
        try:
            self.dead_letter.parent.mkdir(parents=True, exist_ok=True)
            with self._lock, self.dead_letter.open("a", encoding="utf-8") as out:
                out.write(line + "\n")
            kept = f"kept in {self.dead_letter}"
        except OSError as exc:
            # Not even the file: the last resort is the log, at ERROR, without
            # the words -- which are the one part that must not be printed.
            log.error(
                "turn %s/%s was lost (%s; the dead-letter file failed too: %s)",
                record.session_id,
                record.turn_index,
                why,
                exc,
            )
            kept = "lost"
        with self._lock:
            self.dead += 1
        count("audit_dead_letter")
        send_alert(
            "audit:dead_letter",
            f"a turn record could not be saved ({why}); {kept}",
            call=record.session_id,
        )
