"""The office voices, English and Urdu, synthesised here and played by the page.

The reply's own script chooses the voice, so an Urdu answer always gets the
Urdu voice, whatever language the question was asked in. What the voice is
given is the reply made speakable -- numbers as words in its language, no
markdown or links -- while the page shows the reply as written, figures in
digits.

A voice that fails leaves the reply on screen and silent rather than failing
the turn: the customer can still read the answer.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from voicebot.assistant.config import AssistantConfig, Language
from voicebot.assistant.language import spoken_language
from voicebot.assistant.text_prep import prepare_for_speech
from voicebot.assistant.tts_client import TtsClient, TtsError
from voicebot.schemas import AudioBuffer
from voicebot.tts.base import TTSEngine
from voicebot.utils.logging import get_logger
from voicebot.utils.registry import register

log = get_logger("tts.endpoints")

RATE = 24000  # what both voices produce
SILENT = np.zeros(0, dtype=np.float32)


@register("tts", "endpoints")
class EndpointsTTSEngine(TTSEngine):
    speaks_client_side = False

    def __init__(self, voices: Mapping[Language, TtsClient]) -> None:
        self.voices = voices

    @classmethod
    def from_config(cls, config: AssistantConfig) -> EndpointsTTSEngine:
        tts = config.tts
        return cls(
            {
                "en": TtsClient(tts.en, "en", tts.timeout_s, tts.retries, tts.trim_silence),
                "ur": TtsClient(tts.ur, "ur", tts.timeout_s, tts.retries, tts.trim_silence),
            }
        )

    def synthesize(self, text: str, voice: str | None = None) -> AudioBuffer:
        """`voice` may name the language; otherwise the text's script decides."""
        language: Language = voice if voice in ("en", "ur") else spoken_language(text)  # type: ignore[assignment]
        speakable = prepare_for_speech(text, language)
        if not speakable:
            return AudioBuffer(samples=SILENT, sample_rate=RATE)
        try:
            speech = self.voices[language].synthesize(speakable)
        except TtsError as exc:
            log.error("the %s voice failed; the reply is shown but not spoken: %s", language, exc)
            return AudioBuffer(samples=SILENT, sample_rate=RATE)
        return AudioBuffer(
            samples=speech.samples.astype(np.float32) / 32768, sample_rate=speech.sample_rate
        )
