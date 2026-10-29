"""Fault injection: the recognisers and voices failing, as the web bot meets them.

httpx's mock transport plays each office service, timing out, rate-limiting,
erroring, answering nonsense or never answering at all. What comes out must
be a transcript, an ASRUnavailable, a voice clip, or silence the page can
cover -- never another exception, and never a wait without end.
"""

from __future__ import annotations

import threading
import time

import httpx
import numpy as np
import pytest

from voicebot.asr.base import ASRUnavailable
from voicebot.asr.endpoints import EndpointsASREngine
from voicebot.assistant.asr_client import AsrClient, AsrError
from voicebot.assistant.asr_client import Transcript as Heard
from voicebot.assistant.config import AsrEndpoint, TtsEndpoint, load_assistant_config
from voicebot.assistant.language import LanguageRouter
from voicebot.assistant.tts_client import Speech, TtsClient, TtsError
from voicebot.errors import ErrorCode
from voicebot.messages import say
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
    def __init__(self) -> None:
        self.down = False
        self.said: list[str] = []

    def synthesize(self, text: str) -> Speech:
        self.said.append(text)
        if self.down:
            raise TtsError("voice down", code=ErrorCode.PROVIDER_TIMEOUT)
        return Speech(np.array([0, 16_384, -32_768], dtype=np.int16), 24_000, text, 900)


def test_a_fallback_sentence_spoken_once_is_still_heard_when_the_voice_dies() -> None:
    voices = {"en": _Voice(), "ur": _Voice()}
    engine = EndpointsTTSEngine(voices)  # type: ignore[arg-type]
    sorry = say("unavailable", "en")
    first = engine.synthesize(sorry)

    voices["en"].down = True
    assert engine.synthesize(sorry).samples.tolist() == first.samples.tolist()


def test_an_ordinary_reply_with_the_voice_down_is_silent_for_the_page_to_cover() -> None:
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
