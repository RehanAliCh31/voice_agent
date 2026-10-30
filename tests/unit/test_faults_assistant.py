"""Fault injection: the hands-free assistant meeting each failure it can meet.

httpx's mock transport plays the model's server: it rate-limits, refuses,
sends errors and garbage in place of a reply, drops the stream partway and
keeps trickling past the reply's limit. The conversation's own fakes, from
the state-machine tests, fail in the voice, the speaker, the microphone and
even the code that apologises. Whatever happens, the turn ends, the customer
hears an apology or a tone, nothing that was interrupted is heard after the
interruption, and nothing waits for ever.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import urllib.error
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
import pytest

from tests.unit.test_assistant_orchestrator import (
    Harness,
    Llm,
    Router,
    Speaker,
    Voice,
    Wake,
    decision,
    frame,
    frames,
)
from voicebot.assistant import __main__ as cli
from voicebot.assistant import audio_io, health, models
from voicebot.assistant.asr_client import AsrError, Transcript
from voicebot.assistant.audio_io import FRAME, AudioDeviceError, Microphone, list_devices
from voicebot.assistant.config import AssistantConfig, Bilingual, LlmConfig, load_assistant_config
from voicebot.assistant.language import Decision, LanguageRouter
from voicebot.assistant.llm_client import ChatClient, LlmError
from voicebot.assistant.orchestrator import FRAME_FAILURE_LIMIT, State, _Event
from voicebot.assistant.player import RATE, Player
from voicebot.assistant.tts_client import Speech, TtsError
from voicebot.errors import ErrorCode, current_call_id
from voicebot.messages import say
from voicebot.utils.observability import error_counts, event_counts


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    slept: list[float] = []
    monkeypatch.setattr("voicebot.utils.resilience.time.sleep", slept.append)
    return slept


@pytest.fixture
def logged() -> Iterator[list[logging.LogRecord]]:
    records: list[logging.LogRecord] = []

    class Keep(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = Keep()
    logging.getLogger("voicebot").addHandler(handler)
    yield records
    logging.getLogger("voicebot").removeHandler(handler)


@pytest.fixture
def config() -> AssistantConfig:
    cfg = load_assistant_config()
    cfg.vad.min_speech_ms = 64
    cfg.vad.end_silence_ms = 160
    cfg.vad.no_speech_timeout_s = 0.32
    cfg.session.follow_up_s = 0.32
    cfg.session.earcon = False
    cfg.session.echo_tail_ms = 0
    return cfg


# --- the model's server ------------------------------------------------------------


def _llm_config(**changes: object) -> LlmConfig:
    base = {
        "url": "http://llm.test/v1/chat/completions",
        "model": "m",
        "system_prompts": Bilingual(en="be brief", ur="مختصر"),
        "retries": 1,
    }
    return LlmConfig(**{**base, **changes})


def _piece(text: str) -> dict[str, Any]:
    return {"choices": [{"delta": {"content": text}}]}


def _event(event: object) -> bytes:
    return f"data: {json.dumps(event)}\n\n".encode()


def _sse(*events: object) -> bytes:
    return b"".join(_event(e) for e in events) + b"data: [DONE]\n\n"


def _chat(handler: Callable[[httpx.Request], httpx.Response], **changes: object) -> ChatClient:
    return ChatClient(_llm_config(**changes), httpx.Client(transport=httpx.MockTransport(handler)))


def test_a_429_from_the_model_waits_as_long_as_it_is_told_then_succeeds(
    no_sleep: list[float],
) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "1.5"})
        return httpx.Response(200, content=_sse(_piece("Fine.")))

    assert "".join(_chat(handler).reply([])) == "Fine."
    assert len(calls) == 2 and no_sleep and no_sleep[0] >= 1.5


@pytest.mark.parametrize("status", [400, 401, 404, 422])
def test_a_4xx_from_the_model_is_never_retried(no_sleep: list[float], status: int) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(status, text="cannot read: what's my balance")

    with pytest.raises(LlmError) as raised:
        list(_chat(handler).reply([]))
    assert len(calls) == 1 and no_sleep == [] and not raised.value.retryable
    assert "balance" not in str(raised.value)  # the body goes to DEBUG only


def test_an_error_sent_in_place_of_the_reply_is_an_llm_error(no_sleep: list[float]) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        error = {"error": {"message": "the conversation is too long", "code": 400}}
        return httpx.Response(200, content=_sse(error))

    with pytest.raises(LlmError) as raised:
        list(_chat(handler).reply([]))
    assert raised.value.code is ErrorCode.INVALID_INPUT and len(calls) == 1


def test_an_error_sent_partway_is_raised_after_what_came_before_and_not_retried(
    no_sleep: list[float],
) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(
            200, content=_sse(_piece("First part. "), {"error": {"message": "overloaded"}})
        )

    heard = []
    with pytest.raises(LlmError):
        for piece in _chat(handler).reply([]):
            heard.append(piece)
    assert heard == ["First part. "] and len(calls) == 1


@pytest.mark.parametrize(
    "payload",
    [
        "[1, 2]",
        '"just text"',
        '{"choices": ["x"]}',
        '{"choices": {"a": 1}}',
        '{"choices": [{"delta": "x"}]}',
        '{"choices": [5]}',
    ],
)
def test_a_malformed_piece_is_an_llm_error_not_a_crash(no_sleep: list[float], payload: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=f"data: {payload}\n\n".encode())

    with pytest.raises(LlmError):
        list(_chat(handler, retries=0).reply([]))


@pytest.mark.parametrize(
    "body", [{"choices": []}, {"choices": None}, [], {"error": {"message": "busy"}}]
)
def test_a_whole_answer_with_no_reply_in_it_is_an_llm_error(
    no_sleep: list[float], body: object
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    with pytest.raises(LlmError):
        list(_chat(handler, stream=False, retries=0).reply([]))


def test_a_reply_that_keeps_coming_is_cut_off_at_its_total_limit(no_sleep: list[float]) -> None:
    now = [0.0]
    calls = []

    class Trickle(httpx.SyncByteStream):
        def __iter__(self) -> Iterator[bytes]:
            for n in range(100):
                now[0] += 1.0  # a second per piece: no one wait is long, the whole is
                yield _event(_piece(f"word{n} "))

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, stream=Trickle())

    client = ChatClient(
        _llm_config(total_timeout_s=5),
        httpx.Client(transport=httpx.MockTransport(handler)),
        clock=lambda: now[0],
    )
    heard = []
    with pytest.raises(LlmError) as raised:
        for piece in client.reply([]):
            heard.append(piece)
    assert raised.value.code is ErrorCode.PROVIDER_TIMEOUT
    assert 3 <= len(heard) <= 6 and len(calls) == 1


def test_closing_a_reply_closes_its_connection_and_close_closes_the_client() -> None:
    closed = threading.Event()

    class Stream(httpx.SyncByteStream):
        def __iter__(self) -> Iterator[bytes]:
            yield _event(_piece("One. "))
            yield _event(_piece("Two. "))

        def close(self) -> None:
            closed.set()

    http = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=Stream()))
    )
    client = ChatClient(_llm_config(), http)
    reply = client.reply([])
    assert next(reply) == "One. "
    reply.close()  # type: ignore[attr-defined]  # what a barge-in does to it
    assert closed.is_set()
    client.close()
    assert http.is_closed


# --- the conversation, when the model fails ---------------------------------------


class DropsMidway:
    """A model that says part of its reply, then loses the connection."""

    def __init__(self, *pieces: str) -> None:
        self.pieces = pieces

    def reply(
        self, messages: list[dict[str, str]], cancel: threading.Event | None = None
    ) -> Iterator[str]:
        yield from self.pieces
        raise LlmError("the language model could not be reached: ReadError")


def test_a_reply_cut_off_partway_fails_the_turn_and_the_apology_follows_it(
    config: AssistantConfig,
) -> None:
    h = Harness(
        config,
        Router(decision("What's my balance?")),
        DropsMidway("Your balance ", "is fine. ", "And your"),  # type: ignore[arg-type]
    )
    h.feed([frame(wake=True)])
    h.say()
    (turn,) = h.o.turns
    assert turn.outcome == "failed" and turn.reply == "Your balance is fine."
    # What had been said, then the apology; never the half sentence.
    assert h.voices["en"].said == ["Your balance is fine.", say("trouble", "en")]
    assert len(h.o.memory) == 0 and h.o.state is State.IDLE


@pytest.mark.parametrize("pieces", [[], ["   "], ["\n", " "]])
def test_a_reply_with_nothing_to_say_fails_the_turn_with_an_apology(
    config: AssistantConfig, pieces: list[str]
) -> None:
    h = Harness(config, Router(decision("Hi")), Llm(pieces))
    h.feed([frame(wake=True)])
    h.say()
    assert h.o.turns[0].outcome == "failed" and h.o.state is State.IDLE
    assert h.voices["en"].said == [say("trouble", "en")]


def test_a_reply_of_nothing_but_reasoning_fails_the_turn(config: AssistantConfig) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse(_piece("<think>let me see"), _piece("</think>")))

    llm = ChatClient(config.llm, httpx.Client(transport=httpx.MockTransport(handler)))
    h = Harness(config, Router(decision("Hi")), llm)  # type: ignore[arg-type]
    h.feed([frame(wake=True)])
    h.say()
    assert h.o.turns[0].outcome == "failed"
    assert h.voices["en"].said == [say("trouble", "en")]


# --- the conversation, when the voice or the speaker fails -------------------------


class BrokenVoice(Voice):
    def synthesize(self, text: str) -> Speech:
        raise RuntimeError("a bug in the voice")


def test_a_voice_that_raises_anything_ends_the_turn_with_a_tone(config: AssistantConfig) -> None:
    h = Harness(config, Router(decision("Hi")), Llm(["Hello."]), en=BrokenVoice())
    h.feed([frame(wake=True)])
    h.say()
    assert h.o._worker is not None and not h.o._worker.is_alive()
    assert h.speaker.clips == [1]  # the tone, tagged with the turn
    assert h.o.state is State.FOLLOW_UP
    assert error_counts().get(("tts", "internal"), 0) >= 1


class StuckVoice(Voice):
    def __init__(self) -> None:
        super().__init__()
        self.release = threading.Event()

    def synthesize(self, text: str) -> Speech:
        self.release.wait(5)
        return super().synthesize(text)


def test_a_voice_that_stops_making_progress_is_gone_on_without(config: AssistantConfig) -> None:
    stuck = StuckVoice()
    h = Harness(config, Router(decision("Hi")), Llm(["Hello."]), en=stuck)
    h.o.voice_stall_s = 0.2
    h.o._phrases[("fallback", "en")] = Speech(np.ones(2400, np.int16), 24000, "sorry", 1)
    h.feed([frame(wake=True)])
    started = time.monotonic()
    h.say()
    assert time.monotonic() - started < 3
    assert h.speaker.clips == [1]  # the prepared apology, alone
    stuck.release.set()
    voice = next(t for t in h.o._threads if t.name == "voice-1")
    voice.join(2)
    assert not voice.is_alive() and h.speaker.clips == [1]  # the late sentence is never heard


def _apology_breaks(*args: object) -> None:
    raise RuntimeError("the apology broke too")


@pytest.mark.parametrize(
    "router",
    [
        lambda: Router(AsrError("both down")),
        lambda: Router(RuntimeError("a bug in the recogniser")),
    ],
)
def test_an_apology_that_itself_fails_still_ends_the_turn(
    config: AssistantConfig, router: Callable[[], Router]
) -> None:
    h = Harness(config, router(), Llm())
    h.o._say = _apology_breaks  # type: ignore[method-assign]
    h.feed([frame(wake=True)])
    h.say()
    assert h.o._worker is not None and not h.o._worker.is_alive()
    assert h.o.turns[0].outcome == "failed" and h.o.state is State.IDLE


def test_an_apology_that_fails_after_the_model_fails_still_ends_the_turn(
    config: AssistantConfig,
) -> None:
    h = Harness(config, Router(decision("Hi")), Llm(LlmError("down")))
    h.o._say = _apology_breaks  # type: ignore[method-assign]
    h.feed([frame(wake=True)])
    h.say()
    assert h.o.turns[0].outcome == "failed" and h.o.state is State.IDLE


class DeadSpeaker(Speaker):
    """A speaker whose stream has died: it takes clips and never finishes them."""

    def wait(self, timeout: float) -> bool:
        self.hold.wait(min(timeout, 0.01))
        return False


class RaisingSpeaker(Speaker):
    """A speaker that knows its stream died and could not be reopened."""

    def wait(self, timeout: float) -> bool:
        raise AudioDeviceError("the speaker stopped playing, and reopening it did not help")


@pytest.mark.parametrize("speaker", [DeadSpeaker, RaisingSpeaker])
def test_waiting_on_a_speaker_that_cannot_finish_is_bounded(
    config: AssistantConfig, speaker: type[Speaker]
) -> None:
    h = Harness(config, Router(decision("Hi")), Llm(["Hello."]))
    h.speaker = speaker()
    h.speaker.orchestrator = h.o
    h.o.player = h.speaker
    h.o.playback_margin_s = 0.2
    h.feed([frame(wake=True)])
    started = time.monotonic()
    h.say()
    assert time.monotonic() - started < 3
    assert h.o.turns[0].outcome == "answered" and h.o.state is State.FOLLOW_UP
    assert h.speaker.stops >= 1  # stopped, so the next turn starts from silence
    assert error_counts().get(("audio", "audio_error"), 0) >= 1


# --- interruption ------------------------------------------------------------------


class Gated:
    """A model that says one sentence, then waits to be let go before the next."""

    def __init__(self) -> None:
        self.gate = threading.Event()
        self.closed = threading.Event()

    def reply(
        self, messages: list[dict[str, str]], cancel: threading.Event | None = None
    ) -> Iterator[str]:
        try:
            yield "Here is the first part. "
            self.gate.wait(5)
            yield "And here is more. "
            yield "And more still."
        finally:
            self.closed.set()


def _until_speaking(h: Harness) -> None:
    h.feed([frame(wake=True)], frames(4, speech=0.9), frames(5))
    deadline = time.monotonic() + 5
    while h.o.state is not State.SPEAKING and time.monotonic() < deadline:
        h.feed(frames(1))
    assert h.o.state is State.SPEAKING


def test_a_barge_in_closes_the_models_reply_and_drops_queued_recognition(
    config: AssistantConfig,
) -> None:
    llm = Gated()
    h = Harness(config, Router(decision("Tell me everything.")), llm)  # type: ignore[arg-type]
    dropped: list[int] = []
    h.router.cancel_pending = lambda: dropped.append(1) or 0  # type: ignore[attr-defined]
    h.speaker.hold.set()
    _until_speaking(h)
    h.feed([frame(wake=True)])  # woken again: in speakers mode, an interruption
    assert h.o.state is State.LISTENING and dropped == [1]
    llm.gate.set()  # the next piece arrives, to a turn that has been interrupted
    assert llm.closed.wait(2)
    h.settle()
    assert h.speaker.clips == [1]


class SlowSecondSentence(Voice):
    def __init__(self) -> None:
        super().__init__()
        self.gate = threading.Event()
        self.second = threading.Event()

    def synthesize(self, text: str) -> Speech:
        if self.said:  # every sentence after the first waits to be let go
            self.second.set()
            self.gate.wait(5)
        return super().synthesize(text)


def test_a_sentence_synthesised_after_a_barge_in_is_never_played(
    config: AssistantConfig,
) -> None:
    voice = SlowSecondSentence()
    h = Harness(
        config,
        Router(decision("Tell me everything.")),
        Llm(["Here is the first part. ", "And here is the second part."]),
        en=voice,
    )
    h.speaker.hold.set()
    _until_speaking(h)
    assert voice.second.wait(2)
    h.feed([frame(wake=True)])
    voice.gate.set()  # the second sentence is ready -- after the stop
    thread = next(t for t in h.o._threads if t.name == "voice-1")
    thread.join(2)
    h.settle()
    assert not thread.is_alive()
    assert voice.said == ["Here is the first part.", "And here is the second part."]
    assert h.speaker.clips == [1]


def test_recognition_queued_for_an_interrupted_turn_is_dropped() -> None:
    release = threading.Event()
    running: list[int] = []

    class Slow:
        timeout_s = 5.0
        retries = 0

        def transcribe(self, wav: bytes, language: str | None) -> Transcript:
            running.append(1)
            release.wait(5)
            return Transcript("late", "en", 5000)

    router = LanguageRouter("auto", Slow(), Slow())  # type: ignore[arg-type]
    first = threading.Thread(target=router.transcribe, args=(b"w", "en"), daemon=True)
    first.start()
    deadline = time.monotonic() + 2
    while len(running) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)  # both of the router's workers are busy with the first turn
    failed: list[AsrError] = []

    def second() -> None:
        try:
            router.transcribe(b"w", "en")
        except AsrError as exc:
            failed.append(exc)

    waiting = threading.Thread(target=second, daemon=True)
    waiting.start()
    while len(router._jobs) < 4 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert router.cancel_pending() == 2
    waiting.join(2)
    assert failed and not waiting.is_alive() and len(running) == 2
    release.set()
    first.join(2)
    router.close()


# --- the loop itself ----------------------------------------------------------------


def test_a_report_that_cannot_be_acted_on_does_not_end_the_loop(config: AssistantConfig) -> None:
    h = Harness(config, Router(), Llm())
    h.feed([frame(wake=True)])
    h.o._events.put(_Event("done", h.o._turn, None))  # no result in it: acting on it raises
    h.o._events.put(_Event("transcribed", h.o._turn, decision("Hi")))
    h.o.handle_events()
    assert h.o.state is State.THINKING  # the next report was still acted on


class WakeThatCannotReset(Wake):
    def reset(self) -> None:
        raise RuntimeError("the model is gone")


def test_a_detector_that_cannot_reset_does_not_keep_a_session_open(
    config: AssistantConfig,
) -> None:
    h = Harness(config, Router(), Llm())
    h.o.wake = WakeThatCannotReset()
    h.feed([frame(wake=True)], frames(11))
    assert h.o.state is State.IDLE


class BrokenVad:
    def __init__(self) -> None:
        self.resets = 0

    def __call__(self, f: np.ndarray) -> float:
        raise RuntimeError("the model fell over")

    def reset(self) -> None:
        self.resets += 1


def test_a_model_failing_on_every_frame_is_logged_sparingly_and_reset(
    config: AssistantConfig, logged: list[logging.LogRecord]
) -> None:
    h = Harness(config, Router(), Llm())
    vad = BrokenVad()
    h.o.vad = vad
    h.feed([frame(wake=True)])
    before = vad.resets
    h.feed(frames(FRAME_FAILURE_LIMIT))
    assert event_counts()["assistant_frame_errors"] == FRAME_FAILURE_LIMIT
    traces = [r for r in logged if r.getMessage().startswith("a frame could not be processed")]
    assert len(traces) == 1
    assert error_counts()[("audio", "audio_error")] == 1
    assert vad.resets == before + 1 and h.o.wake.resets == 1  # type: ignore[attr-defined]
    assert h.o.state is State.LISTENING


def test_every_turn_runs_under_its_conversations_own_id(config: AssistantConfig) -> None:
    seen: list[str | None] = []

    class Recording(Router):
        def transcribe(self, wav: bytes, session_language: Any) -> Decision:
            seen.append(current_call_id())
            return super().transcribe(wav, session_language)

    h = Harness(
        config,
        Recording(decision("Hi"), decision("And?"), decision("Hello again")),
        Llm(["Hello."], ["Yes."], ["Hi."]),
    )
    h.feed([frame(wake=True)])
    h.say()
    h.say()
    h.feed(frames(11))  # the conversation ends
    h.feed([frame(wake=True)])
    h.say()
    assert seen[0] is not None and seen[0].startswith("wake-") and seen[1] == seen[0]
    assert seen[2] is not None and seen[2] != seen[0]


def test_shutdown_waits_for_the_turns_threads_but_not_for_ever(config: AssistantConfig) -> None:
    release = threading.Event()

    class Hangs(Router):
        def transcribe(self, wav: bytes, session_language: Any) -> Decision:
            release.wait(5)
            raise AsrError("too late")

    h = Harness(config, Hangs(), Llm())
    h.o.shutdown_wait_s = 0.2
    h.feed([frame(wake=True)], frames(4, speech=0.9), frames(5))
    started = time.monotonic()
    h.o.shutdown()
    assert time.monotonic() - started < 2
    assert h.o._worker is not None and h.o._worker.is_alive()
    release.set()
    h.o._worker.join(2)
    assert not h.o._worker.is_alive() and h.speaker.clips == []  # no apology after shutdown


def test_a_microphone_that_dies_ends_the_loop_with_its_error(config: AssistantConfig) -> None:
    h = Harness(config, Router(), Llm())
    stopped = []

    class DeadMic:
        def start(self) -> None:
            pass

        def read(self, timeout: float) -> np.ndarray | None:
            raise AudioDeviceError("the microphone stopped sending audio")

        def stop(self) -> None:
            stopped.append(1)

    with pytest.raises(AudioDeviceError):
        h.o.run(DeadMic(), threading.Event())
    assert stopped == [1]


# --- the sound card -------------------------------------------------------------------


class FakeStream:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.callback = kwargs["callback"]
        self.active = self.closed = self.fail_stop = False
        self.refuse_start = False

    def start(self) -> None:
        if self.refuse_start:
            raise RuntimeError("Invalid sample rate")
        self.active = True

    def stop(self) -> None:
        self.active = False
        if self.fail_stop:
            raise RuntimeError("PortAudio said no")

    def close(self) -> None:
        self.closed = True


class FakeSoundDevice:
    def __init__(self, refuse_start: tuple[int, ...] = ()) -> None:
        self.refuse_start = refuse_start
        self.streams: list[FakeStream] = []

    def query_hostapis(self) -> list[dict[str, Any]]:
        return [{"name": "MME"}]

    def query_devices(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "Headset",
                "hostapi": 0,
                "max_input_channels": 1,
                "max_output_channels": 2,
                "default_samplerate": 48000.0,
            }
        ]

    def _open(self, **kwargs: Any) -> FakeStream:
        stream = FakeStream(**kwargs)
        stream.refuse_start = kwargs["samplerate"] in self.refuse_start
        self.streams.append(stream)
        return stream

    def InputStream(self, **kwargs: Any) -> FakeStream:
        return self._open(**kwargs)

    def OutputStream(self, **kwargs: Any) -> FakeStream:
        return self._open(**kwargs)


def test_a_stream_that_will_not_start_is_closed_before_the_next_rate_is_tried() -> None:
    sd = FakeSoundDevice(refuse_start=(24000,))
    player = Player(backend=sd)
    player.start()
    assert sd.streams[0].closed and sd.streams[1].active and player.rate == 48000
    sd = FakeSoundDevice(refuse_start=(16000,))
    Microphone(backend=sd).start()
    assert sd.streams[0].closed and sd.streams[1].kwargs["samplerate"] == 48000


def test_a_speaker_whose_stream_dies_is_reopened_once_then_reported() -> None:
    sd = FakeSoundDevice()
    player = Player(backend=sd)
    player.start()
    player.enqueue(np.ones(2400, np.int16), RATE)
    sd.streams[0].active = False  # the device went away
    assert player.wait(0.01) is False
    assert len(sd.streams) == 2 and sd.streams[0].closed and player.reopens == 1
    sd.streams[1].active = False  # and the new stream died too, never having played
    with pytest.raises(AudioDeviceError, match="speaker"):
        player.wait(0.01)


def test_a_microphone_that_goes_deaf_is_reopened_once_then_raises() -> None:
    sd = FakeSoundDevice()
    mic = Microphone(backend=sd)
    mic.start()
    sd.streams[0].active = False
    assert mic.read(0.01) is None and len(sd.streams) == 2 and mic.reopens == 1
    sd.streams[1].callback(np.zeros((FRAME, 1), np.int16), FRAME, None, None)  # it hears again
    assert mic.read(0.01) is not None
    sd.streams[1].active = False  # it recovered, so it may be reopened again
    assert mic.read(0.01) is None and len(sd.streams) == 3
    sd.streams[2].active = False  # but not twice without hearing anything in between
    with pytest.raises(AudioDeviceError, match="microphone"):
        mic.read(0.01)


def test_a_microphone_that_stops_calling_back_is_taken_for_dead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sd = FakeSoundDevice()
    mic = Microphone(backend=sd)
    mic.start()
    monkeypatch.setattr(audio_io, "STALL_S", -1.0)  # any time at all without audio is too long
    assert mic.read(0.01) is None and mic.reopens == 1


def test_glitches_a_stream_reports_are_counted() -> None:
    sd = FakeSoundDevice()
    mic = Microphone(backend=sd)
    mic.start()
    player = Player(backend=sd)
    player.start()
    overflow = SimpleNamespace(input_overflow=True)
    sd.streams[0].callback(np.zeros((FRAME, 1), np.int16), FRAME, None, overflow)
    sd.streams[1].callback(
        np.zeros((480, 1), np.int16), 480, None, SimpleNamespace(output_underflow=True)
    )
    counts = event_counts()
    assert counts["audio_input_overflow"] == 1 and counts["audio_output_underflow"] == 1
    assert mic.overflows == 1


def test_a_stream_that_fails_to_stop_is_still_closed() -> None:
    sd = FakeSoundDevice()
    mic = Microphone(backend=sd)
    mic.start()
    player = Player(backend=sd)
    player.start()
    for stream in sd.streams:
        stream.fail_stop = True
    mic.stop()
    player.close()
    assert all(stream.closed for stream in sd.streams)


def test_a_sound_system_that_cannot_list_its_devices_is_an_audio_device_error() -> None:
    class PaError(Exception):
        pass

    class Broken(FakeSoundDevice):
        PortAudioError = PaError

        def query_devices(self) -> list[dict[str, Any]]:
            raise PaError("Error querying device -1")

    with pytest.raises(AudioDeviceError, match="could not list"):
        list_devices(Broken())


# --- the model files, the endpoint check and the command line -----------------------


def test_a_download_that_fails_once_is_retried_and_written_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    data = b"the model"
    monkeypatch.setattr(models, "SHA256", {"melspectrogram.onnx": hashlib.sha256(data).hexdigest()})
    calls = []

    def fetch(url: str, timeout: float = 60.0) -> bytes:
        calls.append(url)
        if len(calls) == 1:
            raise urllib.error.URLError("connection reset by peer")
        return data

    monkeypatch.setattr(models, "_fetch", fetch)
    path = tmp_path / "melspectrogram.onnx"
    models._save(path, models._download("https://release/melspectrogram.onnx"))
    assert path.read_bytes() == data and len(calls) == 2
    assert no_sleep == [models.RETRY_PAUSE_S]
    assert [p.name for p in tmp_path.iterdir()] == ["melspectrogram.onnx"]  # no temporary left


def test_a_download_refused_outright_is_not_retried(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    def fetch(url: str, timeout: float = 60.0) -> bytes:
        raise urllib.error.HTTPError(url, 404, "Not Found", hdrs=None, fp=None)  # type: ignore[arg-type]

    monkeypatch.setattr(models, "_fetch", fetch)
    with pytest.raises(models.ModelDownloadError, match="404"):
        models._download("https://release/missing.onnx")
    assert no_sleep == []


def test_a_write_that_is_interrupted_leaves_the_old_file_and_no_half_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "custom_wake_word.onnx"  # no pinned hash
    path.write_bytes(b"the old, whole file")

    def interrupted(source: str, target: Path) -> None:
        raise OSError("the disk is full")

    monkeypatch.setattr(models.os, "replace", interrupted)
    with pytest.raises(OSError):
        models._save(path, b"the new file")
    assert path.read_bytes() == b"the old, whole file"
    assert [p.name for p in tmp_path.iterdir()] == ["custom_wake_word.onnx"]


@pytest.mark.parametrize(
    "body", [[{"id": "m"}], "models", {"data": "m"}, {"data": [1, None, {"id": 5}]}]
)
def test_a_model_list_of_another_shape_is_no_models_not_a_crash(body: object) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        return httpx.Response(200, json=body)

    check = health.check_service(
        "LLM",
        "http://llm.test/v1/chat/completions",
        "m",
        httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert check.ok and check.warning and "no models listed" in check.detail


class _Voice:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.closed = False

    def synthesize(self, text: str) -> Speech:
        if self.error is not None:
            raise self.error
        return Speech(np.ones(2400, np.int16), 24000, text, 1)

    def close(self) -> None:
        self.closed = True


def test_say_with_the_voice_down_is_one_line_and_a_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    voices = {"en": _Voice(TtsError("the en voice answered 503")), "ur": _Voice()}
    monkeypatch.setattr(cli, "_voices", lambda config: voices)
    assert cli.main(["say", "hello"]) == 1
    out = capsys.readouterr().out
    assert "Could not synthesise it: the en voice answered 503" in out
    assert "Traceback" not in out and all(v.closed for v in voices.values())


def test_say_with_no_speaker_is_one_line_and_a_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def no_speaker(self: Player) -> None:
        raise AudioDeviceError("could not open the speaker: no device")

    monkeypatch.setattr(cli, "_voices", lambda config: {"en": _Voice(), "ur": _Voice()})
    monkeypatch.setattr(Player, "start", no_speaker)
    assert cli.main(["say", "hello"]) == 1
    assert "Could not play it: could not open the speaker" in capsys.readouterr().out


def test_setup_that_cannot_download_is_one_line_and_a_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def down(config: AssistantConfig) -> list[models.ModelFile]:
        raise models.ModelDownloadError("could not download https://release/x: timed out")

    monkeypatch.setattr(cli.models, "download", down)
    assert cli.main(["setup"]) == 1
    out = capsys.readouterr().out
    assert "Could not set up the models: could not download" in out and "Traceback" not in out


def test_ctrl_c_while_starting_up_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def interrupted(config: AssistantConfig) -> list[health.Check]:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.health, "check_all", interrupted)
    hook = threading.excepthook
    assert cli.main(["run"]) == 130
    assert "Stopped." in capsys.readouterr().out
    assert threading.excepthook is hook  # put back as it was found


def test_a_thread_that_dies_is_logged_and_counted(
    monkeypatch: pytest.MonkeyPatch, logged: list[logging.LogRecord]
) -> None:
    def boom() -> None:
        raise ValueError("a bug")

    monkeypatch.setattr(threading, "excepthook", cli._log_thread_crash)
    thread = threading.Thread(target=boom, name="phrases")
    thread.start()
    thread.join()
    assert any(r.getMessage() == "phrases died: ValueError" for r in logged)
    assert error_counts()[("assistant", "internal")] == 1
