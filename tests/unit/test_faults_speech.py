"""Fault injection: the recognisers and voices failing, as the web bot meets them.

httpx's mock transport plays each office service, timing out, rate-limiting,
erroring, answering nonsense or never answering at all. What comes out must
be a transcript, an ASRUnavailable, a voice clip, or silence the page can
cover -- never another exception, and never a wait without end.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import httpx
import numpy as np
import pytest

from voicebot import profile
from voicebot.asr.base import ASRUnavailable
from voicebot.asr.endpoints import EndpointsASREngine
from voicebot.assistant.asr_client import AsrClient, AsrError
from voicebot.assistant.asr_client import Transcript as Heard
from voicebot.assistant.config import AsrEndpoint, TtsEndpoint, load_assistant_config
from voicebot.assistant.language import LanguageRouter
from voicebot.assistant.tts_client import Speech, TtsClient, TtsError
from voicebot.errors import ErrorCode
from voicebot.messages import say
from voicebot.profile import AgentProfile
from voicebot.schemas import AudioBuffer
from voicebot.tts.endpoints import EndpointsTTSEngine
from voicebot.utils.resilience import BreakerState, breaker

CONFIG = load_assistant_config()
QWEN = AsrEndpoint(url="http://en.test/v1/audio/transcriptions", model="qwen3-asr")
WHISPER = AsrEndpoint(
    url="http://ur.test/v1/audio/transcriptions",
    model="whisper-urdu",
    response_format="verbose_json",
)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    slept: list[float] = []
    monkeypatch.setattr("voicebot.utils.resilience.time.sleep", slept.append)
    return slept


def _http(handler) -> httpx.Client:  # type: ignore[no-untyped-def]
    return httpx.Client(transport=httpx.MockTransport(handler))


def _speech(seconds: float = 1.0) -> AudioBuffer:
    t = np.linspace(0, seconds, int(16_000 * seconds), endpoint=False, dtype=np.float32)
    return AudioBuffer(
        samples=(0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sample_rate=16_000
    )


def _wav() -> bytes:
    from voicebot.assistant.asr_client import to_wav

    return to_wav(np.full(2400, 1000, dtype=np.int16), 24_000)


# --- the recognisers -----------------------------------------------------------


def test_a_recogniser_timeout_is_retried_with_backoff(no_sleep: list[float]) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, json={"text": "Hello."})

    assert AsrClient(QWEN, "en", 5, 1, _http(handler)).transcribe(b"w", "en").text == "Hello."
    assert len(calls) == 2 and len(no_sleep) == 1


def test_a_429_from_a_recogniser_is_honoured(no_sleep: list[float]) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "0.5"})
        return httpx.Response(200, json={"text": "ok"})

    AsrClient(QWEN, "en", 5, 1, _http(handler)).transcribe(b"w", "en")
    assert no_sleep[0] >= 0.5


def test_a_connection_dropped_mid_answer_is_an_asr_error_of_the_right_kind() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.RemoteProtocolError("peer closed connection", request=request)

    with pytest.raises(AsrError) as raised:
        AsrClient(QWEN, "en", 5, 1, _http(handler)).transcribe(b"w", "en")
    assert raised.value.code is ErrorCode.PROVIDER_UNAVAILABLE


def test_the_response_body_is_not_logged_above_debug(caplog: pytest.LogCaptureFixture) -> None:
    # A recogniser's error body can quote what the customer said.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="could not decode: my account number is 1234")

    with pytest.raises(AsrError) as raised:
        AsrClient(QWEN, "en", 5, 0, _http(handler)).transcribe(b"w", "en")
    assert "account number" not in str(raised.value)


def test_a_failing_recogniser_is_skipped_and_the_other_decides_alone() -> None:
    def down(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    def up(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"text": "آپ کا بیلنس", "language": "ur", "segments": [{"avg_logprob": -0.1}]}
        )

    router = LanguageRouter(
        "auto", AsrClient(QWEN, "en", 5, 0, _http(down)), AsrClient(WHISPER, "ur", 5, 0, _http(up))
    )
    for _ in range(breaker("asr-en").failures):
        assert router.transcribe(b"w", "en").language == "ur"
    assert breaker("asr-en").state is BreakerState.OPEN
    # With its circuit open the failing one is not even asked, and the turn still works.
    assert router.transcribe(b"w", "en").text == "آپ کا بیلنس"
    router.close()


def test_a_recogniser_that_hangs_past_its_timeout_cannot_hold_the_turn() -> None:
    release = threading.Event()

    class Hangs:
        timeout_s = 0.05
        retries = 0

        def transcribe(self, wav: bytes, language: str | None) -> Heard:
            release.wait(5)
            return Heard("late", "en", 5000)

    class Answers:
        timeout_s = 0.05
        retries = 0

        def transcribe(self, wav: bytes, language: str | None) -> Heard:
            return Heard("آپ کا بیلنس", "ur", 300, "ur", -0.1)

    router = LanguageRouter("auto", Hangs(), Answers())  # type: ignore[arg-type]
    router.wait_margin_s = 0.1
    started = time.monotonic()
    decision = router.transcribe(b"w", "en")
    assert time.monotonic() - started < 1.0
    assert decision.language == "ur"
    release.set()
    router.close()


def test_both_recognisers_down_is_asr_unavailable_with_its_kind() -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("no route", request=request)

    router = LanguageRouter(
        "auto",
        AsrClient(QWEN, "en", 5, 0, _http(timeout)),
        AsrClient(WHISPER, "ur", 5, 0, _http(timeout)),
    )
    with pytest.raises(ASRUnavailable):
        EndpointsASREngine(router, CONFIG.asr).transcribe(_speech())
    router.close()


def test_the_conversations_language_is_the_tie_break_not_another_callers() -> None:
    def nothing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"text": ""})

    router = LanguageRouter(
        "auto",
        AsrClient(QWEN, "en", 5, 0, _http(nothing)),
        AsrClient(WHISPER, "ur", 5, 0, _http(nothing)),
    )
    engine = EndpointsASREngine(router, CONFIG.asr)
    assert engine.transcribe(_speech(), language="ur").language == "ur"
    # A second caller, in English, is not given the first caller's Urdu.
    assert engine.transcribe(_speech(), language="en").language == "en"
    router.close()


# --- the voices --------------------------------------------------------------------


class _Voice:
    def __init__(self, voice: str = "test") -> None:
        self.down = False
        self.said: list[str] = []
        self.endpoint = TtsEndpoint(url="http://t/x", model="m", voice=voice, first_chunk_words=4)

    def synthesize(self, text: str) -> Speech:
        self.said.append(text)
        if self.down:
            raise TtsError("voice down", code=ErrorCode.PROVIDER_TIMEOUT)
        return Speech(
            np.array([0, 16_384, -32_768], dtype=np.int16),
            24_000,
            text,
            900,
            self.endpoint.identity(),
        )


def test_a_fallback_sentence_spoken_once_is_still_heard_when_the_voice_dies() -> None:
    voices = {"en": _Voice(), "ur": _Voice()}
    engine = EndpointsTTSEngine(voices)  # type: ignore[arg-type]
    sorry = say("unavailable", "en")
    first = engine.synthesize(sorry)

    voices["en"].down = True
    assert engine.synthesize(sorry).samples.tolist() == first.samples.tolist()


def test_an_ordinary_reply_with_the_voice_down_is_silent_from_the_engine() -> None:
    voices = {"en": _Voice(), "ur": _Voice()}
    voices["en"].down = True
    audio = EndpointsTTSEngine(voices).synthesize("Your balance is $19,020.00.")  # type: ignore[arg-type]
    assert len(audio.samples) == 0


def test_warming_records_the_apologies_in_both_languages() -> None:
    voices = {"en": _Voice(), "ur": _Voice()}
    engine = EndpointsTTSEngine(voices)  # type: ignore[arg-type]
    assert engine.warm(("unavailable",)) == 2
    voices["ur"].down = True
    assert len(engine.synthesize(say("unavailable", "ur"), voice="ur").samples) == 3


def test_a_voice_that_answers_with_an_empty_clip_is_a_failure() -> None:
    from voicebot.assistant.asr_client import to_wav

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=to_wav(np.zeros(0, dtype=np.int16), 24_000))

    with pytest.raises(TtsError, match="no sound"):
        TtsClient(
            TtsEndpoint(url="http://t/x", model="m"), "en", 5, 0, client=_http(handler)
        ).synthesize("Hi")


def test_a_voice_timeout_then_success() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, content=_wav())

    speech = TtsClient(
        TtsEndpoint(url="http://t/x", model="m"), "en", 5, 1, trim=False, client=_http(handler)
    ).synthesize("Hi")
    assert len(speech.samples) == 2400


def test_engines_release_their_connections() -> None:
    closed: list[str] = []

    class Closes(_Voice):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

        def close(self) -> None:
            closed.append(self.name)

    EndpointsTTSEngine({"en": Closes("en"), "ur": Closes("ur")}).close()  # type: ignore[arg-type]
    assert sorted(closed) == ["en", "ur"]


# --- the bot's own sentences, kept; a reply in pieces ------------------------------


def test_a_kept_recording_survives_a_restart_in_the_same_voice(tmp_path: Path) -> None:
    sorry = say("unavailable", "en")
    EndpointsTTSEngine({"en": _Voice(), "ur": _Voice()}, tmp_path).synthesize(sorry)  # type: ignore[arg-type]

    down = {"en": _Voice(), "ur": _Voice()}
    down["en"].down = True
    restarted = EndpointsTTSEngine(down, tmp_path)  # type: ignore[arg-type]
    assert len(restarted.synthesize(sorry).samples) == 3  # from disk, the voice never asked
    assert restarted.kept("unavailable", "en") is not None


def test_a_kept_recording_is_never_played_as_another_voice(tmp_path: Path) -> None:
    EndpointsTTSEngine({"en": _Voice(), "ur": _Voice()}, tmp_path).warm(("unavailable",))
    other = EndpointsTTSEngine({"en": _Voice("male"), "ur": _Voice("male")}, tmp_path)  # type: ignore[arg-type]
    assert other.kept("unavailable", "en") is None


def test_a_kept_recording_is_never_played_in_the_other_gender(tmp_path: Path) -> None:
    EndpointsTTSEngine({"en": _Voice(), "ur": _Voice()}, tmp_path).warm(("busy",))
    previous = profile.use(AgentProfile(gender="male", names={"en": "Omar", "ur": "عمر"}))
    try:
        male = EndpointsTTSEngine({"en": _Voice(), "ur": _Voice()}, tmp_path)  # type: ignore[arg-type]
        assert male.kept("busy", "ur") is None
    finally:
        profile.use(previous)


def test_a_recording_the_backup_made_is_not_kept_as_the_main_voice() -> None:
    class Backup(_Voice):
        def synthesize(self, text: str) -> Speech:
            speech = super().synthesize(text)
            return Speech(speech.samples, speech.sample_rate, text, 1, "the backup")

    engine = EndpointsTTSEngine({"en": Backup(), "ur": Backup()})  # type: ignore[arg-type]
    engine.synthesize(say("unavailable", "en"))
    assert engine.kept("unavailable", "en") is None


def test_a_reply_is_cut_short_first_then_into_sentences() -> None:
    engine = EndpointsTTSEngine({"en": _Voice(), "ur": _Voice()})  # type: ignore[arg-type]
    pieces = engine.pieces("Your savings balance is $19,020.00. Anything else today?", "en")
    assert pieces[0] == "Your savings balance is"  # the voice's first_chunk_words: 4
    assert " ".join(pieces) == "Your savings balance is $19,020.00. Anything else today?"


def test_recording_the_bots_sentences_waits_while_a_caller_is_answered() -> None:
    voices = {"en": _Voice(), "ur": _Voice()}
    engine = EndpointsTTSEngine(voices)  # type: ignore[arg-type]
    busy = iter([True, True, False] + [False] * 10)
    asked: list[bool] = []

    def turn_running() -> bool:
        running = next(busy)
        asked.append(running)
        return running

    assert engine.warm(("unavailable",), busy=turn_running) == 2
    # It asked, waited through two answers in progress, then recorded.
    assert asked[:3] == [True, True, False]
    assert len(voices["en"].said) == 1


def test_a_caller_who_never_finishes_does_not_stop_the_recordings_for_ever() -> None:
    voices = {"en": _Voice(), "ur": _Voice()}
    engine = EndpointsTTSEngine(voices)  # type: ignore[arg-type]
    assert engine.warm(("unavailable",), busy=lambda: True, give_way_s=0.5) == 2
