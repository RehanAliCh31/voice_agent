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

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from voicebot.agent.executor import ToolExecutor
from voicebot.agent.loop import Conversation
from voicebot.bank.base import Bank
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


class SessionStore(ABC):
    """Reads and writes one session."""

    @abstractmethod
    def get(self, session_id: str) -> Session:
        """Return the session for `session_id`, creating it if new."""

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

    def get(self, session_id: str) -> Session:
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

    def put(self, session_id: str, session: Session) -> None:
        self._entries[session_id] = _Entry(session=session, touched=time.monotonic())
        self._evict_overflow()

    def drop(self, session_id: str) -> None:
        self._entries.pop(session_id, None)

    def __len__(self) -> int:
        return len(self._entries)

    # --- eviction ------------------------------------------------------------

    def _evict_expired(self) -> None:
        cutoff = time.monotonic() - self.ttl_seconds
        stale = [key for key, entry in self._entries.items() if entry.touched < cutoff]
        for key in stale:
            del self._entries[key]
        if stale:
            log.info("evicted %d expired session(s)", len(stale))

    def _evict_overflow(self) -> None:
        """Drop the least recently touched sessions once over the cap."""
        excess = len(self._entries) - self.max_sessions
        if excess <= 0:
            return
        oldest = sorted(self._entries, key=lambda key: self._entries[key].touched)[:excess]
        for key in oldest:
            del self._entries[key]
        log.warning("session cap reached, evicted %d least recent", excess)
