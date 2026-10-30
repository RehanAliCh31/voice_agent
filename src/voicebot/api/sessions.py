"""Where a conversation lives between requests.

A session owns its message history and its pending action. It does **not** own
a bank: there is one bank per process, because this is one customer's
assistant and a phone and a laptop looking at the same account is the
realistic case rather than a bug.

The pending action is per session, though. Two tabs mid-confirmation must not
be able to confirm each other's transfer.

Behind an interface because the in-memory store is correct for one process and
wrong for two. Swapping in Redis means writing another SessionStore, not
touching a single route.
"""

from __future__ import annotations

import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from voicebot.agent.executor import ToolExecutor
from voicebot.agent.loop import Conversation
from voicebot.bank.base import Bank
from voicebot.pipeline.spoken import SpokenReply
from voicebot.utils.logging import get_logger

log = get_logger("api.sessions")


@dataclass
class Session:
    """Everything one caller's conversation owns.

    Named `Session` rather than `Conversation` because `agent.Conversation` is
    the message list; two classes sharing one name across two modules is a bug
    waiting to be written.
    """

    conversation: Conversation
    executor: ToolExecutor
    # One turn at a time. Two requests for one conversation -- a double
    # click on Yes, Enter pressed in the PIN box -- would otherwise run side
    # by side on the same history and the same pending action, and could
    # carry out one transfer twice.
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    started: float = field(default_factory=time.monotonic)
    # How this conversation is going: turns taken, turns in a row that ended
    # in a fallback, the last reply, and whether its end has been announced.
    turns: int = 0
    failures: int = 0
    last_reply: str = ""
    warned: bool = False
    # The rest of the last reply, being spoken a piece at a time. A new turn
    # cancels it, so nothing of an old answer is heard over a new one.
    speaking: SpokenReply | None = None
    # The last turn's result, for a caller that records it later -- the live
    # session, once it knows how much of the reply was heard.
    last_result: Any = None

    def minutes(self) -> float:
        return (time.monotonic() - self.started) / 60


class SessionStore(ABC):
    """Reads and writes one session."""

    @abstractmethod
    def get(self, session_id: str) -> Session:
        """Return the session for `session_id`, creating it if new."""

    def find(self, session_id: str) -> Session | None:
        """The session, if it exists; never makes one."""
        raise NotImplementedError

    @abstractmethod
    def put(self, session_id: str, session: Session) -> None:
        """Persist the session."""

    @abstractmethod
    def drop(self, session_id: str) -> None:
        """Forget a session."""

    @abstractmethod
    def __len__(self) -> int:
        """How many sessions are live."""

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex


@dataclass
class _Entry:
    session: Session
    touched: float


@dataclass
class InMemorySessionStore(SessionStore):
    """Process-local sessions with a TTL and a hard cap.

    Both limits exist because this is reachable from a browser. Without a TTL
    an abandoned conversation lives forever; without a cap, enough distinct
    session ids will exhaust memory. Neither is hypothetical for anything with
    a public URL.
    """

    bank: Bank
    ttl_seconds: float = 1800.0
    max_sessions: int = 1000
    _entries: dict[str, _Entry] = field(default_factory=dict)
    # Requests are served on many threads at once, and eviction walks the
    # whole dict: without a lock, one request adding a session while another
    # evicts raised "dictionary changed size during iteration" -- a 500.
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def get(self, session_id: str) -> Session:
        with self._lock:
            self._evict_expired()
            entry = self._entries.get(session_id)
            if entry is None:
                session = Session(
                    conversation=Conversation(session_id=session_id),
                    executor=ToolExecutor(self.bank, session_id=session_id),
                )
                self._entries[session_id] = _Entry(session=session, touched=time.monotonic())
                self._evict_overflow()
                return session
            entry.touched = time.monotonic()
            return entry.session

    def find(self, session_id: str) -> Session | None:
        with self._lock:
            entry = self._entries.get(session_id)
            return entry.session if entry is not None else None

    def put(self, session_id: str, session: Session) -> None:
        """Mark the session used. One that was ended mid-turn stays ended."""
        with self._lock:
            entry = self._entries.get(session_id)
            if entry is None:
                log.debug("not keeping a session that was ended while its turn ran")
                return
            entry.session = session
            entry.touched = time.monotonic()

    def drop(self, session_id: str) -> None:
        with self._lock:
            entry = self._entries.pop(session_id, None)
        if entry is not None:
            _hang_up(entry.session)

    def __len__(self) -> int:
        with self._lock:
            cutoff = time.monotonic() - self.ttl_seconds
            return sum(1 for entry in self._entries.values() if entry.touched >= cutoff)

    # --- eviction ------------------------------------------------------------

    def _evict_expired(self) -> None:
        cutoff = time.monotonic() - self.ttl_seconds
        stale = [key for key, entry in self._entries.items() if entry.touched < cutoff]
        for key in stale:
            _hang_up(self._entries.pop(key).session)
        if stale:
            log.info("evicted %d expired session(s)", len(stale))

    def _evict_overflow(self) -> None:
        """Drop the least recently touched sessions once over the cap."""
        excess = len(self._entries) - self.max_sessions
        if excess <= 0:
            return
        oldest = sorted(self._entries, key=lambda key: self._entries[key].touched)[:excess]
        for key in oldest:
            _hang_up(self._entries.pop(key).session)
        log.warning("session cap reached, evicted %d least recent", excess)


def _hang_up(session: Session) -> None:
    """The caller has gone: what was left of their last reply is not made."""
    if session.speaking is not None:
        session.speaking.cancel()
        session.speaking = None
