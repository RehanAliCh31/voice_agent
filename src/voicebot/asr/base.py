"""ASR stage interface and a dependency-free fake."""

from __future__ import annotations

from abc import ABC, abstractmethod

from voicebot.errors import ProviderUnavailable
from voicebot.schemas import AudioBuffer, Segment, Transcript
from voicebot.utils.registry import register


class ASRUnavailable(ProviderUnavailable):
    """The recogniser could not be reached. Not the same as hearing nothing."""

    default_component = "asr"


class ASREngine(ABC):
    """Turns audio into text."""

    @abstractmethod
    def transcribe(self, audio: AudioBuffer, language: str | None = None) -> Transcript:
        """Transcribe one utterance.

        `language` is the conversation's own language, for an engine that can
        use it as a tie-break. Passed per call, never kept: the engine is
        shared by every caller, and one caller's Urdu is not another's.
        """

    def close(self) -> None:
        """Release what the engine holds open. Called once, at shutdown."""


@register("asr", "fake")
class FakeASREngine(ASREngine):
    """Returns a fixed string. Used by tests so no model is ever downloaded."""

    def __init__(self, fixed_text: str = "what is my balance", confidence: float = 1.0) -> None:
        self.fixed_text = fixed_text
        self.confidence = confidence

    def transcribe(self, audio: AudioBuffer, language: str | None = None) -> Transcript:
        return Transcript(
            text=self.fixed_text,
            confidence=self.confidence,
            duration_s=audio.duration_s,
            segments=(Segment(text=self.fixed_text, start_s=0.0, end_s=audio.duration_s),),
        )
