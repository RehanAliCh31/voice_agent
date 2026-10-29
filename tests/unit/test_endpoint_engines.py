"""The web bot's office engines, the recognisers and the voices, with fakes for the network.

How the language is decided, and how each voice is spoken to, is tested with
the assistant. These are the seams the web bot adds: audio from the page in,
a transcript that names its language out, and every reply read by the voice
of its own script -- or shown silent when that voice fails.
"""

from __future__ import annotations

import io
import math
import wave

import numpy as np
import pytest

from voicebot.asr.base import ASRUnavailable
from voicebot.asr.endpoints import EndpointsASREngine
from voicebot.assistant.asr_client import AsrError
from voicebot.assistant.asr_client import Transcript as Heard
from voicebot.assistant.config import load_assistant_config
from voicebot.assistant.language import LanguageRouter
from voicebot.assistant.tts_client import Speech, TtsError
from voicebot.schemas import AudioBuffer
from voicebot.tts.endpoints import EndpointsTTSEngine

CONFIG = load_assistant_config()

URDU = "میرے چیکنگ اکاؤنٹ میں کتنے پیسے ہیں"
URDU_BY_QWEN = Heard("मेरे चेकिंग अकाउंट में कितने पैसे हैं", "en", 300)
URDU_BY_WHISPER = Heard(URDU, "ur", 600, "ur", -0.1)
ENGLISH = Heard("What is my savings balance?", "en", 300)
ENGLISH_BY_WHISPER = Heard("What is my savings balance", "ur", 500, "en", -0.2)


def _speech(seconds: float = 1.0, rate: int = 16_000) -> AudioBuffer:
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False, dtype=np.float32)
    return AudioBuffer(
        samples=(0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sample_rate=rate
    )


# --- hearing -----------------------------------------------------------------


class _Recogniser:
    """One office recogniser: says one thing or fails, and keeps what it was sent."""

    def __init__(self, result: Heard | Exception) -> None:
        self.result = result
        self.sent: list[bytes] = []

    def transcribe(self, wav: bytes, language: str | None) -> Heard:
        self.sent.append(wav)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _router(en: Heard | Exception, ur: Heard | Exception) -> LanguageRouter:
    return LanguageRouter("auto", _Recogniser(en), _Recogniser(ur))  # type: ignore[arg-type]


def test_urdu_speech_comes_back_in_urdu_and_says_so() -> None:
    heard = EndpointsASREngine(_router(URDU_BY_QWEN, URDU_BY_WHISPER), CONFIG.asr).transcribe(
        _speech()
    )

    assert (heard.text, heard.language) == (URDU, "ur")
    # Whisper's own average log-probability, as the confidence.
    assert heard.confidence == pytest.approx(math.exp(-0.1))


def test_english_speech_comes_back_in_english() -> None:
    heard = EndpointsASREngine(_router(ENGLISH, ENGLISH_BY_WHISPER), CONFIG.asr).transcribe(
        _speech()
    )

    # qwen3-asr reports no confidence, and is not doubted for it.
    assert (heard.text, heard.language, heard.confidence) == (ENGLISH.text, "en", 1.0)


def test_the_recognisers_are_sent_16_khz_whatever_the_page_recorded() -> None:
    router = _router(ENGLISH, ENGLISH_BY_WHISPER)
    EndpointsASREngine(router, CONFIG.asr).transcribe(_speech(seconds=0.5, rate=48_000))

    for recogniser in (router.en, router.ur):
        with wave.open(io.BytesIO(recogniser.sent[0])) as wav:  # type: ignore[attr-defined]
            assert (wav.getframerate(), wav.getnframes(), wav.getsampwidth()) == (16_000, 8_000, 2)


def test_a_stock_phrase_from_a_moment_of_noise_is_nothing_said() -> None:
    engine = EndpointsASREngine(
        _router(Heard("Okay.", "en", 200), Heard("", "ur", 200, "en", -0.3)), CONFIG.asr
    )

    heard = engine.transcribe(_speech(seconds=0.3))

    assert (heard.text, heard.segments) == ("", ())


def test_the_same_words_said_at_length_are_kept() -> None:
    engine = EndpointsASREngine(
        _router(Heard("Okay.", "en", 200), Heard("", "ur", 200, "en", -0.3)), CONFIG.asr
    )

    assert engine.transcribe(_speech(seconds=1.5)).text == "Okay."


def test_both_recognisers_down_is_unavailable_not_silence() -> None:
    engine = EndpointsASREngine(_router(AsrError("en down"), AsrError("ur down")), CONFIG.asr)

    with pytest.raises(ASRUnavailable, match="en down; ur down"):
        engine.transcribe(_speech())


def test_when_nothing_is_heard_the_conversations_language_stands() -> None:
    # Passed in by the caller, not remembered by the engine: the engine is
    # shared by every conversation, and one caller's Urdu is not another's.
    engine = EndpointsASREngine(_router(Heard("", "en", 1), Heard("", "ur", 1)), CONFIG.asr)

    assert engine.transcribe(_speech(), language="ur").language == "ur"
    assert engine.transcribe(_speech(), language="en").language == "en"


# --- speaking ----------------------------------------------------------------


class _Voice:
    """One office voice: a fixed three samples, or a failure, and what it was asked."""

    def __init__(self, fails: bool = False) -> None:
        self.fails = fails
        self.said: list[str] = []

    def synthesize(self, text: str) -> Speech:
        self.said.append(text)
        if self.fails:
            raise TtsError("voice down")
        return Speech(np.array([0, 16_384, -32_768], dtype=np.int16), 24_000, text, 900)


def _voices(failing: str | None = None) -> dict[str, _Voice]:
    return {"en": _Voice(failing == "en"), "ur": _Voice(failing == "ur")}


def test_an_urdu_reply_is_read_by_the_urdu_voice_with_its_figures_as_urdu_words() -> None:
    voices = _voices()
    EndpointsTTSEngine(voices).synthesize("آپ کے چیکنگ اکاؤنٹ میں $2,483.50 ہیں۔")  # type: ignore[arg-type]

    (said,) = voices["ur"].said
    assert voices["en"].said == []
    assert "ڈالر" in said and not any(c.isdigit() for c in said)


def test_an_english_reply_is_read_by_the_english_voice() -> None:
    voices = _voices()
    EndpointsTTSEngine(voices).synthesize("Your savings balance is $19,020.00.")  # type: ignore[arg-type]

    (said,) = voices["en"].said
    assert voices["ur"].said == []
    assert "dollars" in said and not any(c.isdigit() for c in said)


def test_the_voice_can_be_named() -> None:
    voices = _voices()
    EndpointsTTSEngine(voices).synthesize("OK", voice="ur")  # type: ignore[arg-type]

    assert voices["ur"].said and not voices["en"].said


def test_the_page_is_given_floats_at_the_voices_own_rate() -> None:
    audio = EndpointsTTSEngine(_voices()).synthesize("Hello.")  # type: ignore[arg-type]

    assert audio.sample_rate == 24_000
    assert audio.samples.dtype == np.float32
    assert audio.samples.tolist() == [0.0, 0.5, -1.0]


def test_a_voice_that_fails_leaves_the_reply_silent_not_the_turn_failed() -> None:
    audio = EndpointsTTSEngine(_voices(failing="ur")).synthesize("آپ کا شکریہ۔")  # type: ignore[arg-type]

    assert len(audio.samples) == 0


def test_nothing_speakable_asks_no_voice() -> None:
    voices = _voices()
    audio = EndpointsTTSEngine(voices).synthesize("**")  # type: ignore[arg-type]

    assert len(audio.samples) == 0
    assert voices["en"].said == voices["ur"].said == []


def test_both_engines_are_built_from_the_assistants_file() -> None:
    tts = EndpointsTTSEngine.from_config(CONFIG)
    asr = EndpointsASREngine.from_config(CONFIG)

    assert set(tts.voices) == {"en", "ur"}
    assert tts.speaks_client_side is False
    assert asr.config == CONFIG.asr
    asr.router.close()
