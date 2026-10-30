"""TTS stage interface and a dependency-free fake."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from voicebot.schemas import AudioBuffer
from voicebot.utils.registry import register


class TTSEngine(ABC):
    """Turns text into speech."""

    # True when the engine returns no audio because the client will speak the
    # words itself. The API publishes this so the page knows which it is
    # getting; every server-side engine leaves it False.
    speaks_client_side: bool = False

    def close(self) -> None:
        """Release what the engine holds open. Called once, at shutdown."""

    @abstractmethod
    def synthesize(self, text: str, voice: str | None = None) -> AudioBuffer:
        """Synthesize one reply."""

    def pieces(self, text: str, voice: str | None = None) -> list[str]:
        """`text` cut into the pieces it is best spoken in, first piece first.

        An engine that synthesises quickly enough says a reply whole; one that
        does not cuts it so the first piece can be heard while the rest is
        still being made.
        """
        return [text]

    def kept(self, key: str, language: str) -> AudioBuffer | None:
        """A recording of one of the bot's own sentences, made earlier, if there is one."""
        return None


@register("tts", "fake")
class FakeTTSEngine(TTSEngine):
    """Returns silence proportional to the text length. Used by tests."""

    def __init__(self, sample_rate: int = 16000, seconds_per_char: float = 0.05) -> None:
        self.sample_rate = sample_rate
        self.seconds_per_char = seconds_per_char

    def synthesize(self, text: str, voice: str | None = None) -> AudioBuffer:
        n = max(1, int(len(text) * self.seconds_per_char * self.sample_rate))
        return AudioBuffer(samples=np.zeros(n, dtype=np.float32), sample_rate=self.sample_rate)
