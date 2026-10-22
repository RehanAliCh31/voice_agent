"""NLU stage interface and a dependency-free fake."""

from __future__ import annotations

from abc import ABC, abstractmethod

from voicebot.schemas import Understanding
from voicebot.utils.registry import register


class NLUEngine(ABC):
    """Turns text into an intent plus its slots."""

    @abstractmethod
    def understand(self, text: str) -> Understanding:
        """Classify one utterance and extract its slots."""


@register("nlu", "fake")
class FakeNLUEngine(NLUEngine):
    """Returns a fixed intent. Used by tests."""

    def __init__(self, fixed_intent: str = "balance", confidence: float = 0.99) -> None:
        self.fixed_intent = fixed_intent
        self.confidence = confidence

    def understand(self, text: str) -> Understanding:
        # Empty ASR output must not be guessed at: it is out of scope by definition.
        if not text.strip():
            return Understanding(
                intent="oos", intent_confidence=0.0, oos_score=1.0, is_out_of_scope=True
            )
        return Understanding(intent=self.fixed_intent, intent_confidence=self.confidence)
