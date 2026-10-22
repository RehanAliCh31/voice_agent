"""The whole turn: audio in, audio out.

This is the only place the five stages meet. Each one is held behind its
interface, so the pipeline is identical whether it is driving fake engines in
a test or RoBERTa and Whisper on a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voicebot.asr.base import ASREngine
from voicebot.asr.normalizer import normalize
from voicebot.dialogue.manager import DialogueManager
from voicebot.nlu.base import NLUEngine
from voicebot.pipeline.audio_io import is_silent, trim_silence
from voicebot.pipeline.latency import LatencyBreakdown, LatencyTracker
from voicebot.schemas import (
    ActionType,
    AudioBuffer,
    BotAction,
    DialogueState,
    Transcript,
    Understanding,
)
from voicebot.tts.base import TTSEngine
from voicebot.utils.logging import get_logger

log = get_logger("pipeline.orchestrator")

NOT_CAUGHT = "I didn't catch that. Could you say it again?"
SILENCE = np.zeros(0, dtype=np.float32)


@dataclass(frozen=True)
class TurnResult:
    """Everything one turn produced, including the parts the UI inspects."""

    transcript: Transcript
    text: str
    understanding: Understanding
    action: BotAction
    audio: AudioBuffer
    latency: LatencyBreakdown


class VoicePipeline:
    """Runs one conversational turn end to end."""

    def __init__(
        self,
        asr: ASREngine,
        nlu: NLUEngine,
        tts: TTSEngine,
        manager: DialogueManager | None = None,
        speak: bool = True,
    ) -> None:
        self.asr = asr
        self.nlu = nlu
        self.tts = tts
        self.manager = manager or DialogueManager()
        self.speak = speak

    def converse(self, audio: AudioBuffer, state: DialogueState) -> TurnResult:
        """Audio in, audio out."""
        tracker = LatencyTracker()

        with tracker.stage("vad"):
            trimmed = trim_silence(audio)
            heard_something = not is_silent(trimmed)

        # Guard here, not only inside the ASR engine. A new engine that forgets
        # to check would otherwise let silence through to the NLU, and running
        # a speech model on room tone is both wasted time and an invitation to
        # hallucinate. Checking first also skips the expensive call entirely.
        if not heard_something:
            return self._nothing_heard(
                Transcript(text="", confidence=0.0, duration_s=audio.duration_s), tracker
            )

        with tracker.stage("asr"):
            transcript = self.asr.transcribe(trimmed)

        with tracker.stage("normalize"):
            text = normalize(transcript.text)

        # Nothing was said. Short-circuit rather than ask the NLU to classify
        # an empty string and the dialogue manager to act on the result.
        if not text:
            return self._nothing_heard(transcript, tracker)

        with tracker.stage("nlu"):
            understanding = self.nlu.understand(text)

        with tracker.stage("dialogue"):
            action = self.manager.handle(text, understanding, state)

        with tracker.stage("tts"):
            reply_audio = self._speak(action.text)

        result = TurnResult(
            transcript=transcript,
            text=text,
            understanding=understanding,
            action=action,
            audio=reply_audio,
            latency=tracker.breakdown(),
        )
        self._log(result)
        return result

    def respond_to_text(self, text: str, state: DialogueState) -> TurnResult:
        """Same turn, skipping the microphone.

        Used by tests and by the API's text endpoint. Going through the same
        path keeps one implementation of the turn rather than two that drift.
        """
        tracker = LatencyTracker()
        transcript = Transcript(text=text, confidence=1.0, duration_s=0.0)

        with tracker.stage("normalize"):
            normalized = normalize(text)

        if not normalized:
            return self._nothing_heard(transcript, tracker)

        with tracker.stage("nlu"):
            understanding = self.nlu.understand(normalized)

        with tracker.stage("dialogue"):
            action = self.manager.handle(normalized, understanding, state)

        with tracker.stage("tts"):
            reply_audio = self._speak(action.text)

        result = TurnResult(
            transcript=transcript,
            text=normalized,
            understanding=understanding,
            action=action,
            audio=reply_audio,
            latency=tracker.breakdown(),
        )
        self._log(result)
        return result

    # --- internals -----------------------------------------------------------

    def _speak(self, text: str) -> AudioBuffer:
        if not self.speak:
            return AudioBuffer(samples=SILENCE, sample_rate=16_000)
        return self.tts.synthesize(text)

    def _nothing_heard(self, transcript: Transcript, tracker: LatencyTracker) -> TurnResult:
        """Silence, or a transcript that normalised away to nothing."""
        action = BotAction(type=ActionType.CLARIFY, text=NOT_CAUGHT)
        with tracker.stage("tts"):
            audio = self._speak(action.text)
        return TurnResult(
            transcript=transcript,
            text="",
            understanding=Understanding(
                intent="oos", intent_confidence=0.0, oos_score=1.0, is_out_of_scope=True
            ),
            action=action,
            audio=audio,
            latency=tracker.breakdown(),
        )

    def _log(self, result: TurnResult) -> None:
        slowest = result.latency.slowest()
        log.info(
            "heard=%r intent=%s conf=%.2f action=%s total=%.0fms slowest=%s",
            result.text,
            result.understanding.intent,
            result.understanding.intent_confidence,
            result.action.type.value,
            result.latency.total_ms,
            f"{slowest[0]} {slowest[1]:.0f}ms" if slowest else "n/a",
        )
