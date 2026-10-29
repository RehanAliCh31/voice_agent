"""The assistant against the real endpoints. Deselected by default: pytest -m live

They need the office office network and the model files from
`python -m voicebot.assistant setup`. Speech is made by the voices themselves,
so no microphone is involved and the runs can be repeated exactly.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import numpy as np
import pytest

from voicebot.assistant import health
from voicebot.assistant.asr_client import AsrClient, to_wav
from voicebot.assistant.config import load_assistant_config
from voicebot.assistant.player import to_rate
from voicebot.assistant.text_prep import prepare_for_speech
from voicebot.assistant.tts_client import TtsClient
from voicebot.assistant.vad import FRAME, SileroVad
from voicebot.assistant.wake_word import WakeWord

pytestmark = pytest.mark.live

CONFIG = load_assistant_config()


def _voice(language: str) -> TtsClient:
    t = CONFIG.tts
    return TtsClient(getattr(t, language), language, t.timeout_s, t.retries)


def _speech16(language: str, text: str) -> np.ndarray:
    speech = _voice(language).synthesize(text)
    return to_rate(speech.samples, speech.sample_rate, 16000)


def _script():  # type: ignore[no-untyped-def]
    path = Path(__file__).resolve().parents[2] / "scripts" / "voice_turn.py"
    spec = importlib.util.spec_from_file_location("voice_turn", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_endpoint_answers() -> None:
    checks = health.check_all(CONFIG)
    assert all(c.ok for c in checks), health.report(checks)


def test_the_wake_word_fires_on_the_phrase_and_not_on_other_speech() -> None:
    def fires(audio: np.ndarray) -> bool:
        wake = WakeWord(CONFIG.wake_word)
        padded = np.concatenate([np.zeros(16000, np.int16), audio, np.zeros(16000, np.int16)])
        return any(
            wake.process(padded[i : i + FRAME]) for i in range(0, len(padded) - FRAME, FRAME)
        )

    assert fires(_speech16("en", "Hey Jarvis."))
    assert not fires(_speech16("en", "Good morning, I would like to check my account."))
    assert not fires(_speech16("ur", "آپ کا کارڈ بلاک کر دیا گیا ہے۔"))


@pytest.mark.parametrize(
    ("language", "text"), [("en", "Hello, can you hear me?"), ("ur", "آپ کا کارڈ بلاک ہے۔")]
)
def test_voice_activity_is_found_in_both_voices(language: str, text: str) -> None:
    vad = SileroVad(CONFIG.vad.model_path)
    audio = np.concatenate(
        [np.zeros(16000, np.int16), _speech16(language, text), np.zeros(16000, np.int16)]
    )
    probs = np.array([vad(audio[i : i + FRAME]) for i in range(0, len(audio) - FRAME, FRAME)])
    first = int(np.argmax(probs > CONFIG.vad.threshold))
    assert probs[:25].max() < 0.2  # the leading second of silence
    assert 0.95 <= first * FRAME / 16000 <= 1.2  # speech found within 200 ms of its start


def test_urdu_amounts_are_spoken_as_the_numbers_they_are() -> None:
    spoken = prepare_for_speech("آپ کا بیلنس 2483 روپے ہے۔", "ur")
    asr = CONFIG.asr
    heard = (
        AsrClient(asr.ur, "ur", asr.timeout_s, asr.retries)
        .transcribe(to_wav(_speech16("ur", spoken)), "ur")
        .text
    )
    assert "2483" in re.sub(r"[،,]", "", heard)


@pytest.mark.parametrize("language", ["en", "ur"])
def test_a_whole_voice_turn(language: str, tmp_path: Path) -> None:
    script = _script()
    CONFIG.session.earcon = False
    result = script.run_turn(CONFIG, script.made_up_input(CONFIG, language), tmp_path, language)
    assert result["outcome"] == "answered" and result["language"] == language
    assert result["reply"] and result["reply_audio_s"] > 1
    assert result["seconds_after_end_of_speech"]["first_audio"] < 10
