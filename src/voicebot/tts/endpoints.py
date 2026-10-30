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

import threading
from collections.abc import Mapping

import numpy as np

from voicebot.assistant.config import AssistantConfig, Language
from voicebot.assistant.language import spoken_language
from voicebot.assistant.text_prep import prepare_for_speech
from voicebot.assistant.tts_client import BackedUpVoice, TtsClient, TtsError, build_voices
from voicebot.messages import messages, say
from voicebot.schemas import AudioBuffer
from voicebot.tts.base import TTSEngine
from voicebot.utils.logging import get_logger
from voicebot.utils.registry import register

log = get_logger("tts.endpoints")

RATE = 24000  # what both voices produce
SILENT = np.zeros(0, dtype=np.float32)


# The fallback sentences worth having ready before they are needed: what the
# customer hears when the service is in trouble -- which is exactly when the
# voice may be too.
WARM_KEYS = ("unavailable", "tool_failed", "speech_unavailable", "internal")


@register("tts", "endpoints")
class EndpointsTTSEngine(TTSEngine):
    speaks_client_side = False

    def __init__(self, voices: Mapping[Language, TtsClient | BackedUpVoice]) -> None:
        self.voices = voices
        # Audio for the bot's own fallback sentences, kept from the last time
        # each was spoken, so an apology is still heard when the voice that
        # would say it is down.
        self._fallbacks: dict[tuple[str, str], AudioBuffer] = {}
        self._lock = threading.Lock()
        self._own = {
            (language, sentences[language])
            for sentences in messages().values()
            for language in ("en", "ur")
        }

    @classmethod
    def from_config(cls, config: AssistantConfig) -> EndpointsTTSEngine:
        return cls(build_voices(config))

    def synthesize(self, text: str, voice: str | None = None) -> AudioBuffer:
        """`voice` may name the language; otherwise the text's script decides."""
        language: Language = voice if voice in ("en", "ur") else spoken_language(text)  # type: ignore[assignment]
        speakable = prepare_for_speech(text, language)
        if not speakable:
            return AudioBuffer(samples=SILENT, sample_rate=RATE)
        key = (language, text.strip())
        try:
            speech = self.voices[language].synthesize(speakable)
        except TtsError:
            # Reported by the client already, with the call it belongs to.
            with self._lock:
                kept = self._fallbacks.get(key)
            if kept is not None:
                log.warning("the %s voice is down; playing the kept recording", language)
                return kept
            log.warning("the %s voice is down; the reply is shown but not spoken", language)
            return AudioBuffer(samples=SILENT, sample_rate=RATE)
        audio = AudioBuffer(
            samples=speech.samples.astype(np.float32) / 32768, sample_rate=speech.sample_rate
        )
        if key in self._own:
            with self._lock:
                self._fallbacks[key] = audio
        return audio

    def warm(self, keys: tuple[str, ...] = WARM_KEYS) -> int:
        """Record the most needed fallback sentences now. Returns how many were kept.

        Run once, in the background, after startup. A voice that is down now
        simply leaves them to be kept the first time they are spoken.
        """
        kept = 0
        for key in keys:
            for language in ("en", "ur"):
                if len(self.synthesize(say(key, language), voice=language).samples):
                    kept += 1
        log.info("kept %d fallback recordings ready", kept)
        return kept

    def close(self) -> None:
        for client in self.voices.values():
            close = getattr(client, "close", None)
            if callable(close):
                close()
