"""A hands-free conversation: the microphone stays open, and turns are found by voice.

One LiveSession per connected page. The page streams the microphone, 16 kHz
16-bit mono, in small chunks; the session finds speech in it with Silero VAD,
decides when the caller has finished, runs the turn -- the same turn the
push-to-talk page takes, with its tools, guards, languages and voices -- and
streams the reply's audio back a piece at a time. While the bot speaks it
keeps listening, and stops for the caller.

The state machine is the one source of truth, and every change of state goes
through `_to`, which refuses any change the table below does not allow:

    idle ──start──▶ listening ──speech──▶ user_speaking ──silence──▶ thinking
                       ▲   │                 ▲      ▲                  │
                       │   └─ long silence:  │      └── speech resumes ┘  (the same turn:
                       │      "still there?" │         before the reply   it is asked again
                       │      (agent_speaking)│        is heard           with all of it)
                       │                     │                           ▼
                       └──── reply played ── agent_speaking ◀── reply ───┘
                                             │
                                             └── speech (barge-in) ──▶ user_speaking

    any state ──stop, hang-up, limit──▶ ended

Three threads touch a session: the connection's, which only hands audio in;
this session's audio thread, which runs the VAD and owns the state; and one
turn at a time, which recognises, answers and sends the reply's pieces. A
turn carries a number and a cancel token. Anything it would say is checked
against the session -- still this turn, still thinking, not cancelled --
under the same lock that a barge-in or a stop takes, so a turn that has been
replaced can never speak over the one that replaced it.

What the bot actually said is what the page reports it played. An answer cut
short by the caller goes into the history, and the record of the turn, as
the part that was heard, marked as interrupted -- so the next answer knows
what the caller has and has not been told.
"""

from __future__ import annotations

import base64
import contextvars
import dataclasses
import queue
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

import numpy as np

from voicebot.asr.base import ASRUnavailable
from voicebot.assistant.config import VadConfig
from voicebot.assistant.vad import (
    FRAME,
    Endpointer,
    NoSpeech,
    SpeechStarted,
    Utterance,
    UtteranceEnded,
)
from voicebot.config import LiveConfig
from voicebot.errors import Cancelled, InternalError, call_context
from voicebot.messages import say
from voicebot.pipeline.audio_io import decode_wav, encode_wav
from voicebot.pipeline.orchestrator import TurnResult
from voicebot.schemas import AudioBuffer
from voicebot.utils.logging import get_logger
from voicebot.utils.observability import count, report
from voicebot.utils.resilience import CancelToken, cancel_context

log = get_logger("live.session")

RATE = 16_000
# The history's note on an answer the caller cut short. For the model, never
# spoken: the reply after it knows what was and was not heard.
INTERRUPTED = " [the customer interrupted here and did not hear the rest]"
# How long a turn that began by interrupting waits for the page to say where
# playback stopped, before working it out from the clock instead.
REPORT_WAIT_S = 0.4
# Speech that recognises to nothing, and was at least this long, is answered
# with "I didn't catch that"; anything shorter was a noise, and is ignored.
SAY_NOT_CAUGHT_MS = 900
_PUNCTUATION = re.compile(r"[^\w\s]")


class State(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    USER_SPEAKING = "user_speaking"
    THINKING = "thinking"
    AGENT_SPEAKING = "agent_speaking"
    ENDED = "ended"


ALLOWED: dict[State, frozenset[State]] = {
    State.IDLE: frozenset({State.LISTENING, State.ENDED}),
    State.LISTENING: frozenset({State.USER_SPEAKING, State.AGENT_SPEAKING, State.ENDED}),
    State.USER_SPEAKING: frozenset(
        {State.THINKING, State.LISTENING, State.AGENT_SPEAKING, State.ENDED}
    ),
    State.THINKING: frozenset(
        {State.AGENT_SPEAKING, State.USER_SPEAKING, State.LISTENING, State.ENDED}
    ),
    State.AGENT_SPEAKING: frozenset({State.LISTENING, State.USER_SPEAKING, State.ENDED}),
    State.ENDED: frozenset(),
}


class Vad(Protocol):
    def __call__(self, frame: np.ndarray) -> float: ...
    def reset(self) -> None: ...


Send = Callable[[dict[str, Any] | bytes], None]


@dataclass
class _Turn:
    number: int
    audio: np.ndarray
    speech_ms: int
    speech_ended_at: float  # when the caller stopped speaking
    silence_ms: int  # waited for, before deciding they had
    token: CancelToken = field(default_factory=CancelToken)
    started_at: float = 0.0
    # The reply it interrupted, if it began by interrupting one.
    interrupted: _Playing | None = None
    produced_audio: bool = False


@dataclass
class _Playing:
    """A reply on its way to, or out of, the caller's speaker."""

    audio_id: str
    turn: int
    pieces: tuple[str, ...]
    result: TurnResult | None
    rest: Any = None  # SpokenReply
    durations: list[float] = field(default_factory=list)
    sent: int = 0  # pieces whose audio went to the page
    sent_all: bool = False
    first_sent_at: float = 0.0
    deadline: float = float("inf")
    paused: bool = False
    paused_at: tuple[int, float] | None = None  # (piece, ms into it), as the page reports
    reported: threading.Event = field(default_factory=threading.Event)
    end_after: bool = False  # the session ends once this has been heard
    turn_index: int | None = None
    recorded: bool = False


class LiveSession:
    """One caller's hands-free conversation, from the first word to the last."""

    def __init__(
        self,
        services: Any,
        session_id: str,
        send: Send,
        vad: Vad,
        config: LiveConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.services = services
        self.session_id = session_id
        self.config = config or services.config.live
        self._send = send
        self._vad = vad
        self.clock = clock
        self.state = State.IDLE
        self._lock = threading.RLock()
        # Held by a turn while it is answered. A turn that replaces another --
        # the caller resumed after a pause -- waits for the first to let go of
        # the conversation, rather than be told "one moment" by take_turn.
        self._answering = threading.Lock()
        cfg = self.config
        ears = VadConfig(
            threshold=cfg.threshold,
            neg_threshold=cfg.neg_threshold,
            pre_roll_ms=cfg.pre_roll_ms,
            min_speech_ms=cfg.min_speech_ms,
            end_silence_ms=cfg.end_silence_ms,
            max_utterance_s=cfg.max_utterance_s,
            no_speech_timeout_s=cfg.idle_prompt_s,
        )
        self._ears = Endpointer(ears)
        self._barge = Endpointer(
            ears.model_copy(
                update={
                    "threshold": cfg.barge_in_threshold,
                    "min_speech_ms": cfg.barge_in_min_speech_ms,
                }
            )
        )
        self._active = self._ears
        self._pending = np.zeros(0, dtype=np.int16)
        self._audio: queue.Queue[bytes | None] = queue.Queue(maxsize=512)
        self._turns = 0
        self._turn: _Turn | None = None
        self._merge: np.ndarray | None = None  # an utterance a continuation joins
        self._playing: _Playing | None = None
        # The reply the caller is speaking over, until their turn settles it.
        self._interrupted: _Playing | None = None
        self._idle_prompted = False
        self._started_at = 0.0
        self._threads: list[threading.Thread] = []
        self._audio_thread: threading.Thread | None = None
        self._speech_onset_at = 0.0
        self._barge_stop_ms: float | None = None
        # One of the bot's own sentences being made, cancelled if the caller
        # speaks first.
        self._saying: CancelToken | None = None
        self.metrics: list[dict[str, Any]] = []
        self.interruptions = 0
        self.false_starts = 0
        self.illegal = 0

    # --- the outside world ---------------------------------------------------

    def start(self) -> None:
        """Begin listening. A second start, or one after the end, does nothing."""
        with self._lock:
            if self.state is not State.IDLE:
                log.info("start ignored: the session is already %s", self.state)
                return
            self._started_at = self.clock()
            self._vad.reset()
            self._listen()
            self._to(State.LISTENING)
        self._audio_thread = threading.Thread(
            target=contextvars.copy_context().run,
            args=(self._audio_loop,),
            name=f"live-audio-{self.session_id[:8]}",
            daemon=True,
        )
        self._audio_thread.start()
        self._spawn(self._connect, "live-connect")

    def feed(self, pcm: bytes) -> None:
        """Microphone audio: 16 kHz, 16-bit, mono, any length. Never blocks the caller."""
        if self.state is State.ENDED:
            return
        try:
            self._audio.put_nowait(pcm)
        except queue.Full:
            # Behind by over ten seconds: drop the oldest. Speech is lost
            # either way, and stale audio would only answer an old question.
            try:
                self._audio.get_nowait()
                self._audio.put_nowait(pcm)
            except (queue.Empty, queue.Full):
                pass
            count("live_audio_dropped")

    def played(self, audio_id: str, piece: int, offset_ms: float) -> None:
        """The page stopped playback here, as it was told to."""
        with self._lock:
            playing = self._playing
            if playing is None or playing.audio_id != audio_id:
                return
            playing.paused_at = (piece, max(0.0, offset_ms))
            playing.reported.set()

    def playback_done(self, audio_id: str) -> None:
        """The page finished playing a reply."""
        with self._lock:
            playing = self._playing
            if playing is None or playing.audio_id != audio_id or playing.paused:
                return
            if not playing.sent_all:
                return  # more pieces are coming; the page ran dry before they did
            self._finished_playing(playing)

    def stop(self, reason: str = "stopped") -> None:
        """End the session: nothing more is heard, said or made."""
        with self._lock:
            if self.state is State.ENDED:
                return
            self._end_turn()
            interrupted, self._interrupted = self._interrupted, None
            if interrupted is not None and interrupted is not self._playing:
                self._record(interrupted, interrupted=True)
            playing = self._playing
            if playing is not None:
                if playing.rest is not None:
                    playing.rest.cancel()
                self._record(playing, interrupted=not playing.sent_all or playing.paused)
                self._playing = None
            self._to(State.ENDED)
            self._send({"type": "ended", "reason": reason})
            log.info(
                "live session ended (%s): %d turn(s), %d interruption(s)",
                reason,
                self._turns,
                self.interruptions,
            )
        try:
            self._audio.put_nowait(None)
        except queue.Full:
            pass

    def close(self, timeout_s: float = 3.0) -> None:
        """Stop, and wait for this session's threads to finish."""
        self.stop("closed")
        finish = time.monotonic() + timeout_s
        for thread in [self._audio_thread, *self._threads]:
            if thread is not None and thread is not threading.current_thread():
                thread.join(max(0.0, finish - time.monotonic()))
        alive = [t.name for t in self._threads if t.is_alive()]
        if alive:
            log.warning("live session closed with work still finishing: %s", alive)

    @property
    def threads_alive(self) -> int:
        return sum(t.is_alive() for t in [self._audio_thread, *self._threads] if t is not None)

    # --- state ----------------------------------------------------------------

    def _to(self, state: State) -> bool:
        """Change state, if the table allows it, and tell the page. Under the lock."""
        if state is self.state:
            return True
        if state not in ALLOWED[self.state]:
            self.illegal += 1
            log.error("live session: %s -> %s is not allowed; ignored", self.state, state)
            return False
        self.state = state
        self._send({"type": "state", "state": state.value})
        return True

    def _listen(self) -> None:
        """Wait for the caller: asking, after a while, whether they are still there."""
        wait = self.config.idle_end_s if self._idle_prompted else self.config.idle_prompt_s
        self._ears.start(wait_s=wait)
        self._active = self._ears

    # --- audio ------------------------------------------------------------------

    def _audio_loop(self) -> None:
        while True:
            try:
                pcm = self._audio.get(timeout=0.25)
            except queue.Empty:
                self._safe_tick()
                continue
            if pcm is None or self.state is State.ENDED:
                return
            samples = np.frombuffer(pcm, dtype="<i2")
            self._pending = np.concatenate([self._pending, samples])
            while len(self._pending) >= FRAME:
                frame, self._pending = self._pending[:FRAME], self._pending[FRAME:]
                try:
                    self._frame(frame)
                except Exception as exc:  # one bad frame must not end the conversation
                    report(InternalError("a live audio frame failed", cause=exc))
            self._safe_tick()

    def _safe_tick(self) -> None:
        try:
            self._tick()
        except Exception as exc:  # the timers failing must not leave the session deaf
            report(InternalError("the live session's timers failed", cause=exc))

    def _frame(self, frame: np.ndarray) -> None:
        prob = self._vad(frame)
        with self._lock:
            if self.state in (State.IDLE, State.ENDED):
                return
            event = self._active.process(frame, prob)
            if self.state is State.USER_SPEAKING and self._active is self._barge:
                self._maybe_confirm_interruption()
            if event is None:
                return
            if isinstance(event, SpeechStarted):
                self._speech_started()
            elif isinstance(event, UtteranceEnded):
                self._utterance_ended(event.utterance)
            elif isinstance(event, NoSpeech):
                self._no_speech()

    def _tick(self) -> None:
        """Timers: a reply nobody said had finished, and the session's own limit."""
        with self._lock:
            if self.state is State.ENDED:
                return
            now = self.clock()
            playing = self._playing
            if (
                self.state is State.AGENT_SPEAKING
                and playing is not None
                and playing.sent_all
                and not playing.paused
                and now > playing.deadline
            ):
                log.info("the page never said the reply finished; going on")
                self._finished_playing(playing)
            turn = self._turn
            if (
                self.state is State.THINKING
                and turn is not None
                and now - turn.started_at > self._turn_limit_s()
            ):
                log.warning("a live turn ran past its limit; letting it go")
                count("live_turns_timed_out")
                turn.started_at = float("inf")  # once
                self._say_then("unavailable", end=False, turn=turn)
            if (
                self._started_at
                and now - self._started_at > self.config.max_session_minutes * 60
                and self.state in (State.LISTENING, State.USER_SPEAKING)
                and self._playing is None
            ):
                log.info("the live session reached its time limit")
                self._say_then("session_ended", end=True)

    def _turn_limit_s(self) -> float:
        """The longest a turn may take: the model's whole budget, and the voice's first piece."""
        resilience = getattr(self.services.config, "resilience", None)
        budget = getattr(resilience, "turn_budget_s", 20.0)
        return float(budget) + 20.0

    def _speech_started(self) -> None:
        self._speech_onset_at = self.clock() - self._active.speech_ms / 1000
        if self.state is State.LISTENING:
            if self._saying is not None:
                self._saying.cancel("the caller spoke first")
                self._saying = None
            self._to(State.USER_SPEAKING)
            self._idle_prompted = False
        elif self.state is State.THINKING:
            # The caller had only paused: this is the same turn. It is asked
            # again, with everything they said, before any of it is answered.
            turn = self._turn
            if turn is not None and not turn.produced_audio:
                log.info("speech resumed before the reply was heard: one turn, not two")
                count("live_continuations")
                self._merge = turn.audio
                self._end_turn()
                self._to(State.USER_SPEAKING)
        elif self.state is State.AGENT_SPEAKING:
            self._barge_in()

    def _barge_in(self) -> None:
        """The caller spoke over the bot: it stops at once, and listens."""
        playing = self._playing
        if playing is None:
            return
        playing.paused = True
        self._interrupted = playing
        self._send({"type": "stop_audio", "audio_id": playing.audio_id, "reason": "barge_in"})
        stop_ms = (self.clock() - self._speech_onset_at) * 1000
        self._to(State.USER_SPEAKING)
        self.interruptions += 1
        count("live_interruptions")
        log.info("barge-in: playback stopped %.0f ms after speech began", stop_ms)
        self._barge_stop_ms = stop_ms

    def _maybe_confirm_interruption(self) -> None:
        """Speech that has gone on longer than a "mm-hmm" did interrupt."""
        playing = self._interrupted
        if (
            playing is not None
            and self._playing is playing
            and self._active.speech_ms > self.config.backchannel_max_ms
        ):
            self._confirm_interruption(playing)

    def _confirm_interruption(self, playing: _Playing) -> None:
        """Nothing more of the reply is made or kept. Under the lock.

        What was heard of it goes into the history when the interrupting
        turn is answered, before its own question does (_finish_interruption).
        """
        if playing.rest is not None:
            playing.rest.cancel()
        if self._playing is playing:
            self._playing = None
            self._send({"type": "discard_audio", "audio_id": playing.audio_id})

    def _utterance_ended(self, utterance: Utterance) -> None:
        if self.state is not State.USER_SPEAKING:
            return
        silence = self._active.silence_ms
        audio = utterance.audio
        if self._merge is not None:
            # A pause, then more: the turn is all of it, the pause included.
            gap = np.zeros(int(RATE * 0.2), dtype=np.int16)
            audio = np.concatenate([self._merge, gap, audio])
            self._merge = None
        interrupted, self._interrupted = self._interrupted, None
        self._turns += 1
        turn = _Turn(
            number=self._turns,
            audio=audio,
            speech_ms=utterance.speech_ms,
            speech_ended_at=self.clock() - silence / 1000,
            silence_ms=silence,
            interrupted=interrupted,
            started_at=self.clock(),
        )
        self._turn = turn
        self._to(State.THINKING)
        # Listening on, for the caller resuming before the reply is heard.
        self._ears.start(wait_s=None)
        self._active = self._ears
        self._spawn(lambda: self._run_turn(turn), f"live-turn-{turn.number}")

    def _no_speech(self) -> None:
        if self.state is not State.LISTENING:
            return
        if self._idle_prompted:
            log.info("nobody has spoken since asking; ending the live session")
            self._say_then("idle_goodbye", end=True)
        else:
            self._idle_prompted = True
            self._say_then("still_there", end=False)

    # --- turns ------------------------------------------------------------------

    def _end_turn(self) -> None:
        """Cancel the turn in progress, if there is one. Under the lock."""
        turn, self._turn = self._turn, None
        if turn is not None:
            turn.token.cancel("the turn was replaced")

    def _current(self, turn: _Turn) -> bool:
        return self._turn is turn and not turn.token.cancelled and self.state is State.THINKING

    def _run_turn(self, turn: _Turn) -> None:
        with call_context(self.session_id), cancel_context(turn.token):
            try:
                self._turn_body(turn)
            except Cancelled:
                log.info("turn %d was replaced before it was answered", turn.number)
            except Exception as exc:  # whatever went wrong, the caller hears a sentence
                log.error("live turn failed: %s", type(exc).__name__, exc_info=exc)
                report(InternalError(f"a live turn failed: {type(exc).__name__}", cause=exc))
                with self._lock:
                    if self._current(turn):
                        self._say_then("internal", end=False, turn=turn)

    def _turn_body(self, turn: _Turn) -> None:
        services = self.services
        conversation = services.sessions.get(self.session_id).conversation
        started = self.clock()
        try:
            transcript = services.asr.transcribe(
                AudioBuffer(turn.audio.astype(np.float32) / 32768, RATE),
                language=conversation.language,
            )
        except ASRUnavailable:
            with self._lock:
                if self._current(turn):
                    self._finish_interruption(turn)
                    self._say_then("speech_unavailable", end=False, turn=turn)
            return
        asr_ms = (self.clock() - started) * 1000
        turn.token.check("the answer")
        text = (transcript.text or "").strip()
        if text:
            self._send(
                {
                    "type": "transcript",
                    "text": text,
                    "language": transcript.language or conversation.language,
                    "turn": turn.number,
                }
            )
        if turn.interrupted is not None and self._is_backchannel(text, turn):
            if self._resume(turn):
                return
        if turn.interrupted is not None:
            # Where playback stopped, from the page -- it says at once, so
            # this rarely waits -- before the history is told.
            turn.interrupted.reported.wait(REPORT_WAIT_S)
        with self._lock:
            if not self._current(turn):
                return
            self._finish_interruption(turn)
        if not _has_words(text):
            self.false_starts += 1
            count("live_false_starts")
            with self._lock:
                if not self._current(turn):
                    return
                if turn.speech_ms >= SAY_NOT_CAUGHT_MS:
                    self._say_then("not_caught", end=False, turn=turn)
                else:
                    log.info("a noise, not words: listening on")
                    self._turn = None
                    self._listen()
                    self._to(State.LISTENING)
            return

        with self._answering:
            turn.token.check("the answer")
            session = services.sessions.get(self.session_id)
            snapshot = (
                [dict(m) for m in session.conversation.messages],
                session.conversation.turn_index,
                session.executor.writes,
            )
            try:
                payload = services.take_turn(
                    self.session_id,
                    lambda s, p: p.respond_to_heard(transcript, s.conversation, asr_ms),
                    record=False,
                )
            except Cancelled:
                self._restore(snapshot)
                raise
            result: TurnResult | None = session.last_result
            if turn.token.cancelled:
                # Replaced while it was answered: its answer is not given, and
                # the history goes back to before it -- unless money moved.
                self._restore(snapshot)
                if result is not None and result.rest is not None:
                    result.rest.cancel()
                return
        self._speak(turn, payload, result, asr_ms)

    def _restore(self, snapshot: tuple[list[dict[str, Any]], int, int]) -> None:
        """Put the conversation back as it was before a turn nobody will hear."""
        messages, turn_index, writes = snapshot
        session = self.services.sessions.get(self.session_id)
        if session.executor.writes != writes:
            log.warning("a replaced turn changed the bank; its history is kept")
            return
        session.conversation.messages[:] = messages
        session.conversation.turn_index = turn_index
        session.executor.cancel()

    def _is_backchannel(self, text: str, turn: _Turn) -> bool:
        if turn.speech_ms > self.config.backchannel_max_ms:
            return False
        said = _PUNCTUATION.sub(" ", text.lower()).split()
        return not said or " ".join(said) in {b.lower() for b in self.config.backchannels}

    def _resume(self, turn: _Turn) -> bool:
        """It was only a "mm-hmm": the reply goes on from where it stopped.

        False when there is nothing left to go on with -- the reply was
        already let go -- and the turn is then answered like any other.
        """
        with self._lock:
            playing = turn.interrupted
            if not self._current(turn) or playing is None or self._playing is not playing:
                return False
            self.interruptions -= 1
            log.info("a backchannel, not an interruption: the reply goes on")
            count("live_backchannels")
            playing.paused = False
            if playing.sent_all:
                # Time again for all of it: the clock ran while it was paused.
                playing.deadline = (
                    self.clock() + sum(playing.durations) + self.config.playback_margin_s
                )
            self._turn = None
            self._send({"type": "resume_audio", "audio_id": playing.audio_id})
            self._to(State.AGENT_SPEAKING)
            self._barge.start(wait_s=None)
            self._active = self._barge
            return True

    def _finish_interruption(self, turn: _Turn) -> None:
        """The reply this turn interrupted is over: only what was heard of it stays. Under the lock.

        Run by the interrupting turn before it asks its own question, so the
        history it asks with already holds the answer as heard -- and the
        last answer in the history is still the interrupted one.
        """
        playing = turn.interrupted
        if playing is None:
            return
        turn.interrupted = None
        self._confirm_interruption(playing)
        if playing.recorded:
            return
        spoken = self._heard(playing)
        if playing.result is not None:
            session = self.services.sessions.get(self.session_id)
            for message in reversed(session.conversation.messages):
                if message.get("role") == "assistant" and not message.get("tool_calls"):
                    message["content"] = f"{spoken}{INTERRUPTED}" if spoken else INTERRUPTED.strip()
                    break
        log.info("the reply was interrupted after %d word(s)", len(spoken.split()))
        self._record(playing, interrupted=True, spoken=spoken)

    def _heard(self, playing: _Playing) -> str:
        """The words of a reply that were played before it was stopped."""
        if playing.paused_at is not None:
            piece, offset_ms = playing.paused_at
        else:
            # The page never said: worked out from when the audio was sent.
            elapsed = max(0.0, self.clock() - playing.first_sent_at)
            piece, offset_ms = 0, elapsed * 1000
            for duration in playing.durations:
                if offset_ms < duration * 1000:
                    break
                offset_ms -= duration * 1000
                piece += 1
        words: list[str] = []
        for index, text in enumerate(playing.pieces):
            if index < piece:
                words += text.split()
            elif index == piece:
                duration = playing.durations[index] if index < len(playing.durations) else 0.0
                part = text.split()
                share = min(1.0, offset_ms / (duration * 1000)) if duration else 0.0
                words += part[: round(len(part) * share)]
                break
        return " ".join(words)

    # --- speaking ----------------------------------------------------------------

    def _speak(self, turn: _Turn, payload: Any, result: TurnResult | None, asr_ms: float) -> None:
        """Say the turn's reply, a piece at a time, unless it has been replaced."""
        audio = result.audio if result is not None else _payload_audio(payload)
        pieces = result.pieces if result is not None and result.pieces else (payload.reply,)
        with self._lock:
            if not self._current(turn):
                if result is not None and result.rest is not None:
                    result.rest.cancel()
                return
            session = self.services.sessions.get(self.session_id)
            playing = _Playing(
                audio_id=payload.audio_id or f"t{turn.number}",
                turn=turn.number,
                pieces=pieces,
                result=result,
                rest=result.rest if result is not None else None,
                end_after=payload.fallback in ("session_ended",),
                turn_index=session.conversation.turn_index,
            )
            turn.produced_audio = True
            body = payload.model_dump(exclude={"audio_b64"})
            body.update({"type": "reply", "turn": turn.number, "audio_id": playing.audio_id})
            self._send(body)
            sent_at = self._begin_playing(playing, audio)
            self._turn = None
        latency = payload.latency_ms or {}
        metrics = {
            "type": "metrics",
            "turn": turn.number,
            "vad_to_endpoint_ms": turn.silence_ms,
            "stt_ms": round(asr_ms),
            "agent_ms": round(latency.get("agent", 0.0)),
            "tts_first_ms": round(latency.get("tts", 0.0)),
            "end_of_speech_to_first_audio_ms": round((sent_at - turn.speech_ended_at) * 1000),
            "interruptions": self.interruptions,
        }
        stop = getattr(self, "_barge_stop_ms", None)
        if stop is not None:
            metrics["barge_in_stop_ms"] = round(stop)
            self._barge_stop_ms = None
        self.metrics.append(metrics)
        self._send(metrics)
        log.info(
            "live turn %d: endpoint %d ms, stt %d ms, agent %d ms, tts %d ms, "
            "first audio %d ms after speech, interruptions %d",
            turn.number,
            metrics["vad_to_endpoint_ms"],
            metrics["stt_ms"],
            metrics["agent_ms"],
            metrics["tts_first_ms"],
            metrics["end_of_speech_to_first_audio_ms"],
            self.interruptions,
        )
        self._stream_rest(playing)

    def _begin_playing(self, playing: _Playing, audio: AudioBuffer | None) -> float:
        """Send the first piece and start listening for barge-in. Under the lock."""
        self._playing = playing
        playing.first_sent_at = self.clock()
        if audio is not None and len(audio.samples):
            self._send_piece(playing, 0, audio)
        else:
            playing.durations.append(0.0)
        if playing.rest is None:
            self._all_sent(playing)
        self._to(State.AGENT_SPEAKING)
        self._barge.start(wait_s=None)
        self._active = self._barge
        return playing.first_sent_at

    def _send_piece(self, playing: _Playing, number: int, audio: AudioBuffer) -> None:
        data = encode_wav(audio)
        playing.durations.append(audio.duration_s)
        playing.sent += 1
        self._send(
            {
                "type": "audio",
                "audio_id": playing.audio_id,
                "piece": number,
                "text": playing.pieces[number] if number < len(playing.pieces) else "",
                "seconds": round(audio.duration_s, 3),
                "last": playing.rest is None or number >= playing.rest.count,
            }
        )
        self._send(data)

    def _stream_rest(self, playing: _Playing) -> None:
        """The reply's other pieces, each as soon as it is made."""
        rest = playing.rest
        if rest is None:
            return
        for number in range(1, rest.count + 1):
            audio = rest.wait(number, timeout=45.0)
            with self._lock:
                if rest.cancelled:
                    # Let go of -- the page has been told to discard it -- even
                    # if this piece was made just before: no more of it goes out.
                    return
                if self._playing is not playing and not playing.paused:
                    return
                if audio is None:
                    self._send({"type": "audio_failed", "audio_id": playing.audio_id})
                    self._all_sent(playing)
                    return
                self._send_piece(playing, number, audio)
        with self._lock:
            self._all_sent(playing)

    def _all_sent(self, playing: _Playing) -> None:
        playing.sent_all = True
        playing.deadline = self.clock() + sum(playing.durations) + self.config.playback_margin_s
        # How many pieces there are: the page plays to the last of them, even
        # one still decoding when this arrives.
        self._send(
            {"type": "audio_end", "audio_id": playing.audio_id, "pieces": playing.sent}
        )

    def _finished_playing(self, playing: _Playing) -> None:
        """The reply was heard to its end. Under the lock."""
        self._playing = None
        self._record(playing, interrupted=False)
        if playing.end_after:
            self._to(State.ENDED)
            self._send({"type": "ended", "reason": "said goodbye"})
            try:
                self._audio.put_nowait(None)
            except queue.Full:
                pass
            return
        self._listen()
        self._to(State.LISTENING)

    def _say_then(self, key: str, end: bool, turn: _Turn | None = None) -> None:
        """One of the bot's own sentences, in its voice and gender. Under the lock.

        Made on another thread -- usually a recording kept from before, but
        it may need the voice -- and said only if, by then, nothing has
        overtaken it: the caller speaking, the turn replaced, the end.
        """
        token = CancelToken()
        if turn is None:
            if self._saying is not None:
                self._saying.cancel("superseded")
            self._saying = token
        self._spawn(lambda: self._say_worker(key, end, token, turn), f"live-say-{key}")

    def _say_worker(self, key: str, end: bool, token: CancelToken, turn: _Turn | None) -> None:
        with call_context(self.session_id):
            try:
                session = self.services.sessions.get(self.session_id)
                language = session.conversation.language
                result = self.services.pipeline(session).say(say(key, language), key=key)
            except Exception as exc:  # the sentence cannot be made: go on without it
                report(InternalError(f"could not say {key}", cause=exc))
                result = None
            with self._lock:
                if token.cancelled or self.state is State.ENDED:
                    return
                if turn is not None and not self._current(turn):
                    return
                if turn is None and self.state is not State.LISTENING:
                    return
                if turn is None:
                    self._saying = None
                else:
                    # Answered with this sentence: whatever it was still doing is not wanted.
                    turn.token.cancel("answered with the bot's own sentence")
                self._turn = None
                text = result.agent.text if result is not None else say(key, language)
                playing = _Playing(
                    audio_id=f"say-{key}-{self._turns}-{int(self.clock() * 1000)}",
                    turn=turn.number if turn is not None else 0,
                    pieces=(result.pieces if result is not None else ()) or (text,),
                    result=None,
                    end_after=end,
                )
                self._send({"type": "say", "key": key, "reply": text, "audio_id": playing.audio_id})
                self._begin_playing(playing, result.audio if result is not None else None)

    def _record(self, playing: _Playing, interrupted: bool, spoken: str | None = None) -> None:
        """The turn's record, now that it is known how much of it was heard."""
        if playing.recorded or playing.result is None:
            playing.recorded = True
            return
        playing.recorded = True
        result = playing.result
        if interrupted:
            heard = spoken if spoken is not None else self._heard(playing)
            result = dataclasses.replace(
                result,
                agent=dataclasses.replace(result.agent, text=f"{heard}{INTERRUPTED}".strip()),
            )
        try:
            self.services.record_turn(self.session_id, playing.turn_index or 0, result)
        except Exception as exc:  # the conversation goes on without its record
            report(InternalError("a live turn could not be recorded", cause=exc))

    # --- background work ------------------------------------------------------

    def _spawn(self, work: Callable[[], None], name: str) -> None:
        self._threads = [t for t in self._threads if t.is_alive()]
        thread = threading.Thread(
            target=contextvars.copy_context().run, args=(work,), name=name, daemon=True
        )
        self._threads.append(thread)
        thread.start()

    def _connect(self) -> None:
        """Open the providers' connections now, so the first turn does not."""
        for part in (self.services.asr, self.services.tts):
            connect = getattr(part, "connect", None)
            if callable(connect):
                try:
                    connect()
                except Exception as exc:  # warming is a nicety; the turn connects anyway
                    log.info("could not warm a connection: %s", type(exc).__name__)


def _payload_audio(payload: Any) -> AudioBuffer | None:
    """The audio a turn's payload carries, for a turn with no result of its own."""
    if not getattr(payload, "audio_b64", None):
        return None
    try:
        return decode_wav(base64.b64decode(payload.audio_b64))
    except Exception:  # unreadable: the reply is still shown
        return None


def _has_words(text: str) -> bool:
    return bool(re.search(r"[^\W_]", text or ""))
