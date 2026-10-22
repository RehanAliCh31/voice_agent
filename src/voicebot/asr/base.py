"""ASR stage interface and a dependency-free fake."""

from __future__ import annotations

from abc import ABC, abstractmethod

from voicebot.schemas import AudioBuffer, Segment, Transcript
from voicebot.utils.registry import register


class ASREngine(ABC):
    """Turns audio into text."""

    @abstractmethod
    def transcribe(self, audio: AudioBuffer) -> Transcript:
        """Transcribe one utterance."""


@register("asr", "fake")
class FakeASREngine(ASREngine):
    """Returns a fixed string. Used by tests so no model is ever downloaded."""

    def __init__(self, fixed_text: str = "what is my balance", confidence: float = 1.0) -> None:
        self.fixed_text = fixed_text
        self.confidence = confidence

    def transcribe(self, audio: AudioBuffer) -> Transcript:
        return Transcript(
            text=self.fixed_text,
            confidence=self.confidence,
            duration_s=audio.duration_s,
            segments=(Segment(text=self.fixed_text, start_s=0.0, end_s=audio.duration_s),),
        )
