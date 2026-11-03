"""Stand-ins for driving a live session with synthetic audio: a page, a VAD, voices."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from voicebot.api.deps import Services, build_services
from voicebot.asr.base import ASREngine, ASRUnavailable
from voicebot.config import AppConfig
from voicebot.live.session import LiveSession
from voicebot.llm.base import FakeLLMClient, LLMClient, Reply
from voicebot.schemas import AudioBuffer, Transcript
from voicebot.tts.base import FakeTTSEngine

RATE = 16_000
CHUNK = 320  # 20 ms, as the page sends it
LOUD, MURMUR = 8000, 1500


def tone(ms: int, amplitude: int = LOUD) -> np.ndarray:
    """Sound the stand-in VAD scores as speech (or, quieter, as unsure)."""
    t = np.arange(int(RATE * ms / 1000)) / RATE
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.int16)


def quiet(ms: int) -> np.ndarray:
    return np.zeros(int(RATE * ms / 1000), dtype=np.int16)


class LevelVad:
    """Speech by loudness: loud is sure speech, a murmur unsure, quiet none."""

    def __init__(self) -> None:
        self.frames = 0

    def __call__(self, frame: np.ndarray) -> float:
        self.frames += 1
        level = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        if level > 3000:
            return 0.95
        if level > 600:
            return 0.6
        return 0.02

    def reset(self) -> None:
        pass


class VoiceVad(LevelVad):
    """Speech at any loudness, as Silero hears it: a voice across the room is speech too."""

    def __call__(self, frame: np.ndarray) -> float:
        self.frames += 1
        level = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        return 0.95 if level > 100 else 0.02


class ScriptedASR(ASREngine):
    """Hears what it is told to, in order; can be slow, or down."""

    def __init__(self, heard: list[tuple[str, str] | Exception], delay_s: float = 0.0) -> None:
        self.heard = list(heard)
        self.delay_s = delay_s
        self.calls: list[float] = []  # seconds of audio each was given
        self._lock = threading.Lock()

    def transcribe(self, audio: AudioBuffer, language: str | None = None) -> Transcript:
        with self._lock:
            self.calls.append(audio.duration_s)
            item = self.heard.pop(0) if self.heard else ("", "en")
        if self.delay_s:
            time.sleep(self.delay_s)
        if isinstance(item, Exception):
            raise item
        text, lang = item
        return Transcript(text=text, confidence=1.0, duration_s=audio.duration_s, language=lang)


class PiecesVoice(FakeTTSEngine):
    """Cuts a reply at each full stop, like the office voices cut it, and records the voice."""

    def __init__(self) -> None:
        super().__init__(sample_rate=16_000, seconds_per_char=0.03)
        self.said: list[tuple[str, str | None]] = []
        self.slow: dict[str, float] = {}  # pieces that take this long to make
        self._lock = threading.Lock()

    def pieces(self, text: str, voice: str | None = None) -> list[str]:
        cut = [p.strip() + "." for p in text.replace("۔", ".").split(".") if p.strip()]
        return cut or [text]

    def synthesize(self, text: str, voice: str | None = None) -> AudioBuffer:
        with self._lock:
            self.said.append((text, voice))
        if text in self.slow:
            time.sleep(self.slow[text])
        return super().synthesize(text, voice)


class SlowLLM(LLMClient):
    """A scripted model that takes its time."""

    def __init__(self, replies: list[Reply], delay_s: float) -> None:
        self.inner = FakeLLMClient(replies)
        self.delay_s = delay_s

    def complete(self, messages, tools, tool_choice="auto"):  # type: ignore[no-untyped-def]
        time.sleep(self.delay_s)
        return self.inner.complete(messages, tools, tool_choice)


def services_for(
    heard: list[tuple[str, str] | Exception],
    replies: list[Reply],
    asr_delay_s: float = 0.0,
    llm: LLMClient | None = None,
    session: dict[str, Any] | None = None,
    resilience: dict[str, Any] | None = None,
    **live: Any,
) -> Services:
    config = AppConfig()
    config = config.model_copy(
        update={
            "live": config.live.model_copy(update={"idle_prompt_s": 600, **live}),
            "session": config.session.model_copy(update=session or {}),
            "resilience": config.resilience.model_copy(update=resilience or {}),
        }
    )
    services = build_services(config)
    services.asr = ScriptedASR(heard, asr_delay_s)
    services.llm = llm or FakeLLMClient(replies)
    services.tts = PiecesVoice()
    services.live_vad_factory = LevelVad
    return services


class Page:
    """What the page does: plays, says when it has finished, says where it was stopped."""

    def __init__(self, auto_finish: bool = True, keep_audio: bool = True) -> None:
        self.events: list[dict[str, Any]] = []
        self.audio: list[bytes] = []
        self.audio_count = 0
        self.keep_audio = keep_audio
        self.auto_finish = auto_finish
        self.session: LiveSession | None = None
        self.stop_at: tuple[int, float] = (0, 200.0)  # where playback is when told to stop
        self._lock = threading.Condition()

    def __call__(self, message: dict[str, Any] | bytes) -> None:
        with self._lock:
            if isinstance(message, bytes):
                self.audio_count += 1
                if self.keep_audio:
                    self.audio.append(message)
            else:
                self.events.append(message)
            self._lock.notify_all()
        if isinstance(message, bytes) or self.session is None:
            return
        kind = message.get("type")
        if kind == "audio_end" and self.auto_finish:
            # Played at once: the page reports it has heard the end.
            threading.Thread(
                target=self.session.playback_done, args=(message["audio_id"],), daemon=True
            ).start()
        elif kind == "stop_audio":
            piece, offset = self.stop_at
            threading.Thread(
                target=self.session.played,
                args=(message["audio_id"], piece, offset),
                daemon=True,
            ).start()

    def kinds(self) -> list[str]:
        with self._lock:
            return [e["type"] for e in self.events]

    def of(self, kind: str) -> list[dict[str, Any]]:
        with self._lock:
            return [e for e in self.events if e["type"] == kind]

    def states(self) -> list[str]:
        return [e["state"] for e in self.of("state")]

    def wait(self, test: Callable[[Page], bool], timeout: float = 5.0) -> bool:
        finish = time.monotonic() + timeout
        with self._lock:
            while not test(self):
                left = finish - time.monotonic()
                if left <= 0:
                    return False
                self._lock.wait(left)
        return True


def open_session(services: Services, page: Page, session_id: str = "live-test") -> LiveSession:
    session = LiveSession(services, session_id, page, services.make_live_vad())
    page.session = session
    session.start()
    return session


def speak(session: LiveSession, *parts: np.ndarray, far: float | None = None) -> None:
    """Feed the session audio as the page streams it: 20 ms at a time.

    `far` is how loud the page says it is playing the bot meanwhile.
    """
    audio = np.concatenate(parts) if parts else np.zeros(0, np.int16)
    for start in range(0, len(audio), CHUNK):
        session.feed(audio[start : start + CHUNK].astype("<i2").tobytes(), far)


def drained(session: LiveSession, timeout: float = 5.0) -> bool:
    """Wait until the session has heard all the audio fed to it.

    `speak` feeds faster than real time; a page plays in real time, so
    anything it reports comes after the audio it overlapped was heard.
    """
    finish = time.monotonic() + timeout
    while session._audio.qsize() and time.monotonic() < finish:
        time.sleep(0.01)
    time.sleep(0.05)  # the last chunk taken is being heard
    return not session._audio.qsize()


def settle(page: Page, state: str, timeout: float = 5.0) -> bool:
    """Wait until the session is in `state`."""
    return page.wait(lambda p: bool(p.states()) and p.states()[-1] == state, timeout)


__all__ = [
    "LOUD",
    "MURMUR",
    "ASRUnavailable",
    "Page",
    "PiecesVoice",
    "ScriptedASR",
    "SlowLLM",
    "VoiceVad",
    "open_session",
    "quiet",
    "services_for",
    "settle",
    "speak",
    "tone",
]
