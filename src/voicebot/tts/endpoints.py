"""The office voices, English and Urdu, synthesised here and played by the page.

The voice is the customer's language -- the turn passes it -- or, for the
bot's own sentences, the text's script. What the voice is given is the reply
made speakable -- numbers as words in its language, no markdown or links --
while the page shows the reply as written, figures in digits.

A reply is cut into pieces (`pieces`): the first is short, so it is heard
while the rest is still being made. Neither voice streams, and each costs
seconds per request, so the first piece is what the customer waits for.

The bot's own sentences are recorded once and kept, on disk as well as in
memory, so an apology is still heard when the voice that would say it is
down, and a restart does not record them all again. A recording is kept
under the voice that made it and the bot's gender: change either and it is
never played again, so a kept recording cannot speak in the wrong voice.

A voice that fails leaves the reply on screen and silent rather than failing
the turn: the customer can still read the answer.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

from voicebot import profile
from voicebot.assistant.asr_client import to_wav
from voicebot.assistant.config import AssistantConfig, Language
from voicebot.assistant.language import spoken_language
from voicebot.assistant.text_prep import SentenceStream, prepare_for_speech, split_long
from voicebot.assistant.tts_client import (
    BackedUpVoice,
    TtsClient,
    TtsError,
    build_voices,
    parse_wav,
)
from voicebot.messages import messages, say
from voicebot.schemas import AudioBuffer
from voicebot.tts.base import TTSEngine
from voicebot.utils.logging import get_logger
from voicebot.utils.registry import register

log = get_logger("tts.endpoints")

RATE = 24000  # what both voices produce
SILENT = np.zeros(0, dtype=np.float32)
CACHE_DIR = Path("data/voice_cache")

# The bot's sentences recorded first, before the rest: what the customer
# hears when the service is in trouble -- which is exactly when the voice
# may be too -- and what the page plays when it cannot reach the service.
FIRST_KEYS = (
    "voice_unavailable",
    "unavailable",
    "tool_failed",
    "speech_unavailable",
    "internal",
    "connection_lost",
    "timed_out",
    "microphone_refused",
    "microphone_unavailable",
    "microphone_lost",
)


@register("tts", "endpoints")
class EndpointsTTSEngine(TTSEngine):
    speaks_client_side = False

    def __init__(
        self,
        voices: Mapping[Language, TtsClient | BackedUpVoice],
        cache_dir: Path | None = None,
    ) -> None:
        self.voices = voices
        self.cache_dir = cache_dir
        # Audio for the bot's own sentences, by (language, sentence), kept from
        # the last time each was spoken, so an apology is still heard when the
        # voice that would say it is down.
        self._fallbacks: dict[tuple[str, str], AudioBuffer] = {}
        self._lock = threading.Lock()
        self.gender = profile.current().gender
        self._own = {
            (language, sentences[language])
            for sentences in messages(self.gender).values()
            for language in ("en", "ur")
        }
        self._load_kept()

    @classmethod
    def from_config(
        cls, config: AssistantConfig, cache_dir: Path | None = CACHE_DIR
    ) -> EndpointsTTSEngine:
        return cls(build_voices(config), cache_dir)

    # --- speaking --------------------------------------------------------------

    def _language(self, text: str, voice: str | None) -> Language:
        return voice if voice in ("en", "ur") else spoken_language(text)  # type: ignore[return-value]

    def pieces(self, text: str, voice: str | None = None) -> list[str]:
        """The first piece cut short -- at a comma, or after the voice's first_chunk_words --
        then pieces growing by its chunk_growth, then whole sentences, none of
        them longer than a voice makes quickly (split_long)."""
        language = self._language(text, voice)
        endpoint = self.voices[language].endpoint
        stream = SentenceStream(
            first_words=endpoint.first_chunk_words, growth=endpoint.chunk_growth
        )
        found = [*stream.feed(text), *stream.flush()]
        cut = [short for piece in found for short in split_long(piece, language)]
        return [piece for piece in cut if prepare_for_speech(piece, language)] or [text]

    def synthesize(self, text: str, voice: str | None = None) -> AudioBuffer:
        """`voice` may name the language; otherwise the text's script decides."""
        language = self._language(text, voice)
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
        # Kept only when the main voice made it: one made by the backup is the
        # right gender, but not the voice the rest of the conversation has.
        if key in self._own and speech.voice == self._identity(language):
            with self._lock:
                self._fallbacks[key] = audio
            self._store(language, key[1], speech.samples, speech.sample_rate)
        return audio

    # --- the bot's own sentences, kept ---------------------------------------------

    def kept(self, key: str, language: str) -> AudioBuffer | None:
        with self._lock:
            return self._fallbacks.get((language, say(key, language)))

    def warm(
        self,
        keys: tuple[str, ...] | None = None,
        busy: Callable[[], bool] | None = None,
        give_way_s: float = 60.0,
    ) -> int:
        """Record the bot's own sentences now, the most needed first. Returns how many are kept.

        Run once, in the background, after startup; sentences already kept
        from an earlier run are not recorded again. A voice that is down now
        simply leaves them to be kept the first time they are spoken.

        Each voice makes one clip at a time, so a recording made while a
        caller waits for an answer delays the answer. While `busy` says a
        turn is running, the next recording waits -- up to `give_way_s` --
        and the caller goes first.
        """
        wanted = list(keys) if keys is not None else self._all_keys()
        for key in wanted:
            for language in ("en", "ur"):
                if self.kept(key, language) is None:
                    waited = 0.0
                    while busy is not None and busy() and waited < give_way_s:
                        time.sleep(0.25)
                        waited += 0.25
                    self.synthesize(say(key, language), voice=language)
        kept = sum(self.kept(k, lang) is not None for k in wanted for lang in ("en", "ur"))
        log.info("%d of %d recordings of the bot's own sentences kept", kept, 2 * len(wanted))
        return kept

    def _all_keys(self) -> list[str]:
        rest = [key for key in messages(self.gender) if key not in FIRST_KEYS]
        return [*FIRST_KEYS, *rest]

    def _identity(self, language: str) -> str:
        return self.voices[language].endpoint.identity()  # type: ignore[index]

    def _path(self, language: str, sentence: str) -> Path | None:
        if self.cache_dir is None:
            return None
        name = f"{self._identity(language)}|{self.gender}|{language}|{sentence}"
        return self.cache_dir / f"{hashlib.sha256(name.encode()).hexdigest()[:24]}.wav"

    def _store(self, language: str, sentence: str, samples: np.ndarray, rate: int) -> None:
        path = self._path(language, sentence)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(to_wav(samples, rate))
        except OSError as exc:  # kept in memory all the same; only the next start loses it
            log.warning("could not keep a recording on disk: %s", exc)

    def _load_kept(self) -> None:
        """The recordings an earlier run kept, for this voice and this gender only."""
        loaded = 0
        for language, sentence in self._own:
            path = self._path(language, sentence)
            if path is None or not path.is_file():
                continue
            try:
                samples, rate = parse_wav(path.read_bytes())
            except (OSError, TtsError) as exc:  # a damaged file is recorded again
                log.warning("a kept recording could not be read: %s", type(exc).__name__)
                continue
            self._fallbacks[(language, sentence)] = AudioBuffer(
                samples=samples.astype(np.float32) / 32768, sample_rate=rate
            )
            loaded += 1
        if loaded:
            log.info("%d kept recordings of the bot's own sentences loaded", loaded)

    def connect(self) -> None:
        """Open each voice's connection now, for a live session about to need it."""
        for client in self.voices.values():
            warm = getattr(client, "warm", None)
            if callable(warm):
                warm()

    def close(self) -> None:
        for client in self.voices.values():
            close = getattr(client, "close", None)
            if callable(close):
                close()
