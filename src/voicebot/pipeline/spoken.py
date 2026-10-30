"""A reply spoken a piece at a time: the first with the turn, the rest as each is made.

Neither office voice streams, and each takes seconds per request, so a reply
synthesised whole kept the customer in silence for all of it -- 4.2 s, and
once 24 s, in the baseline. Now the turn answers as soon as its first piece
is ready, and the rest are made one after another in the background while
the first plays. The page asks for each piece in turn and is given it as
soon as it exists.

Pieces belong to their turn. A new turn, or the caller leaving, cancels what
was left of the last one: nothing still being made can be heard over the
next answer. A piece the voice could not make ends the reply there -- the
rest would be fragments -- and the page says the voice has failed.
"""

from __future__ import annotations

import contextvars
import secrets
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from voicebot.errors import InternalError
from voicebot.schemas import AudioBuffer
from voicebot.utils.logging import get_logger
from voicebot.utils.observability import report

log = get_logger("pipeline.spoken")

# Shared by every conversation. The voices take one request at a time, so
# more workers than this would only queue at the server. Made when first
# needed, and again after a shutdown, for a service started a second time
# in the same process.
_POOL: ThreadPoolExecutor | None = None
_POOL_LOCK = threading.Lock()


def _pool() -> ThreadPoolExecutor:
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="voice")
        return _POOL


def shutdown() -> None:
    """Stop making pieces: at service shutdown."""
    global _POOL
    with _POOL_LOCK:
        pool, _POOL = _POOL, None
    if pool is not None:
        pool.shutdown(wait=False, cancel_futures=True)


class SpokenReply:
    """The pieces of one reply after its first, made in the background, in order."""

    def __init__(self, pieces: list[str], speak: Callable[[str], AudioBuffer]) -> None:
        self.id = secrets.token_urlsafe(8)
        self._speak = speak
        self._pieces = list(pieces)
        self._audio: list[AudioBuffer | None] = [None] * len(self._pieces)
        self._done = [threading.Event() for _ in self._pieces]
        self._cancelled = threading.Event()
        self._lock = threading.Lock()
        self._next = 0
        self._running = False
        self._failed = False  # the voice failed: nothing after that point is made
        self._start()

    @property
    def count(self) -> int:
        """How many pieces there are after the first."""
        with self._lock:
            return len(self._pieces)

    def add(self, piece: str) -> None:
        """One more piece at the end: a sentence the conversation adds after the reply."""
        with self._lock:
            self._pieces.append(piece)
            self._audio.append(None)
            self._done.append(threading.Event())
            if self._cancelled.is_set() or self._failed:
                self._done[-1].set()
                return
        self._start()

    def wait(self, number: int, timeout: float) -> AudioBuffer | None:
        """Piece `number` (1 is the one after the first) once made; None if it never will be."""
        with self._lock:
            if not 1 <= number <= len(self._pieces):
                raise IndexError(number)
            done = self._done[number - 1]
        if not done.wait(timeout):
            log.warning("piece %d of a reply was not ready in %.0fs", number, timeout)
            return None
        with self._lock:
            # A cancelled reply gives nothing out, made or not: it is not to be heard.
            return None if self._cancelled.is_set() else self._audio[number - 1]

    def cancel(self) -> None:
        """Make and give out nothing more: the turn has been replaced, or the caller has gone."""
        self._cancelled.set()
        with self._lock:
            for event in self._done:
                event.set()

    # --- the worker ------------------------------------------------------------

    def _start(self) -> None:
        with self._lock:
            if self._running or self._next >= len(self._pieces):
                return
            self._running = True
        try:
            _pool().submit(contextvars.copy_context().run, self._run)
        except RuntimeError:  # the pool is shut down: the service is stopping
            self.cancel()

    def _run(self) -> None:
        while True:
            with self._lock:
                if self._cancelled.is_set() or self._next >= len(self._pieces):
                    self._running = False
                    return
                index, piece = self._next, self._pieces[self._next]
            try:
                audio = self._speak(piece)
            except Exception as exc:  # whatever the voice did, the waiting page must hear of it
                report(
                    InternalError(
                        f"a piece of the reply could not be spoken: {type(exc).__name__}",
                        component="tts",
                        cause=exc,
                    )
                )
                audio = None
            with self._lock:
                if audio is not None and len(audio.samples):
                    self._audio[index] = audio
                    self._done[index].set()
                    self._next += 1
                    continue
                # The voice failed on this piece: it and the rest go unheard,
                # and the page is told rather than left waiting.
                log.warning(
                    "the voice failed on piece %d of a reply; the rest is not spoken", index + 1
                )
                for event in self._done[index:]:
                    event.set()
                self._next = len(self._pieces)
                self._failed = True
                self._running = False
                return
