"""The office services against the real endpoints. Deselected by default: pytest -m live

They need the office office network and the model file from
`python -m voicebot.assistant setup`. Speech is made by the voices themselves,
so no microphone is involved and the runs can be repeated exactly.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from voicebot.assistant import health, models
from voicebot.assistant.asr_client import AsrClient, to_rate, to_wav
from voicebot.assistant.config import VadConfig, load_assistant_config
from voicebot.assistant.text_prep import prepare_for_speech
from voicebot.assistant.tts_client import TtsClient
from voicebot.assistant.vad import FRAME, SileroVad

pytestmark = pytest.mark.live

CONFIG = load_assistant_config()


def _voice(language: str) -> TtsClient:
    t = CONFIG.tts
    return TtsClient(getattr(t, language), language, t.timeout_s, t.retries)


def _speech16(language: str, text: str) -> np.ndarray:
    speech = _voice(language).synthesize(text)
    return to_rate(speech.samples, speech.sample_rate, 16000)


def test_every_endpoint_answers() -> None:
    checks = health.check_all(CONFIG)
    assert all(c.ok for c in checks), health.report(checks)


@pytest.mark.parametrize(
    ("language", "text"), [("en", "Hello, can you hear me?"), ("ur", "آپ کا کارڈ بلاک ہے۔")]
)
def test_voice_activity_is_found_in_both_voices(language: str, text: str) -> None:
    vad = SileroVad(models.SILERO_PATH)
    audio = np.concatenate(
        [np.zeros(16000, np.int16), _speech16(language, text), np.zeros(16000, np.int16)]
    )
    probs = np.array([vad(audio[i : i + FRAME]) for i in range(0, len(audio) - FRAME, FRAME)])
    first = int(np.argmax(probs > VadConfig().threshold))
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
