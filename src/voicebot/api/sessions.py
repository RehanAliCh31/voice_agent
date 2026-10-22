"""Where a conversation lives between requests.

A conversation is the dialogue state AND the account it is acting on. Both
have to be session-scoped: a shared bank would let one browser tab spend
another's balance, and a per-request bank would reset the balance after every
transfer, so the money would never actually move.

Behind an interface because the in-memory store is correct for one process
and wrong for two. Swapping in Redis means writing another SessionStore, not
touching a single route.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from voicebot.dialogue.mock_backend import MockBank
from voicebot.schemas import DialogueState
from voicebot.utils.logging import get_logger

log = get_logger("api.sessions")


@dataclass
class Conversation:
    """Everything one caller's conversation owns."""

    state: DialogueState
    bank: MockBank


class SessionStore(ABC):
    """Reads and writes one conversation."""

    @abstractmethod
    def get(self, session_id: str) -> Conversation:
        """Return the conversation for `session_id`, creating it if new."""

    @abstractmethod
    def put(self, session_id: str, conversation: Conversation) -> None:
        """Persist the conversation."""

    @abstractmethod
    def drop(self, session_id: str) -> None:
        """Forget a conversation."""

    @abstractmethod
    def __len__(self) -> int:
        """How many conversations are live."""

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex


@dataclass
class _Entry:
    conversation: Conversation
    touched: float


@dataclass
class InMemorySessionStore(SessionStore):
    """Process-local sessions with a TTL and a hard cap.

    Both limits exist because this is reachable from a browser. Without a TTL
    an abandoned conversation lives forever; without a cap, enough distinct
    session ids will exhaust memory. Neither is hypothetical for anything with
    a public URL.
    """

    ttl_seconds: float = 1800.0
    max_sessions: int = 1000
    _entries: dict[str, _Entry] = field(default_factory=dict)

    def get(self, session_id: str) -> Conversation:
        self._evict_expired()
        entry = self._entries.get(session_id)
        if entry is None:
            conversation = Conversation(state=DialogueState(session_id=session_id), bank=MockBank())
            self._entries[session_id] = _Entry(conversation=conversation, touched=time.monotonic())
            self._evict_overflow()
            return conversation
        entry.touched = time.monotonic()
        return entry.conversation

    def put(self, session_id: str, conversation: Conversation) -> None:
        self._entries[session_id] = _Entry(conversation=conversation, touched=time.monotonic())
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
        """Drop the least recently touched conversations once over the cap."""
        excess = len(self._entries) - self.max_sessions
        if excess <= 0:
            return
        oldest = sorted(self._entries, key=lambda key: self._entries[key].touched)[:excess]
        for key in oldest:
            del self._entries[key]
        log.warning("session cap reached, evicted %d least recent", excess)
