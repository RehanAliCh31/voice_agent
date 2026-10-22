"""The whole turn: audio in, audio out.

This is the only place the four stages meet. Each one is held behind its
interface, so the pipeline is identical whether it is driving fakes in a test
or Whisper and a 27B model over the network.

v1 ran a normalisation stage between the ASR and the NLU, rewriting Whisper's
output into the lowercased, punctuation-free form the trained model expected.
There is no such stage now: casing and punctuation are signal to a language
model, and stripping them would be deliberate damage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from voicebot.agent.loop import Agent, AgentResult, Conversation
from voicebot.asr.base import ASREngine
from voicebot.pipeline.audio_io import is_silent, trim_silence
from voicebot.pipeline.latency import LatencyBreakdown, LatencyTracker
from voicebot.schemas import AudioBuffer, Transcript
from voicebot.tts.base import TTSEngine
from voicebot.utils.logging import get_logger

log = get_logger("pipeline.orchestrator")

NOT_CAUGHT = "I didn't catch that. Could you say it again?"
SILENCE = np.zeros(0, dtype=np.float32)

# Whisper returns punctuation-only strings on room tone -- "...", ".", "?!".
# Anything with no letter or digit in it carries no request.
_HAS_CONTENT = re.compile(r"[A-Za-z0-9]")


@dataclass(frozen=True)
class TurnResult:
    """Everything one turn produced, including the parts the UI inspects."""

    transcript: Transcript
    text: str
    agent: AgentResult
    audio: AudioBuffer
    latency: LatencyBreakdown


class VoicePipeline:
    """Runs one conversational turn end to end."""

    def __init__(
        self,
        asr: ASREngine,
        agent: Agent,
        tts: TTSEngine,
        speak: bool = True,
    ) -> None:
        self.asr = asr
        self.agent = agent
        self.tts = tts
        self.speak = speak

    def converse(self, audio: AudioBuffer, conversation: Conversation) -> TurnResult:
        """Audio in, audio out."""
        tracker = LatencyTracker()

        with tracker.stage("vad"):
            trimmed = trim_silence(audio)
            heard_something = not is_silent(trimmed)

        # Guard here, not only inside the ASR engine. A new engine that forgets
        # to check would otherwise let silence through, and running a speech
        # model on room tone is both wasted time and an invitation to
        # hallucinate. Checking first also skips the expensive call entirely.
        if not heard_something:
            return self._nothing_heard(
                Transcript(text="", confidence=0.0, duration_s=audio.duration_s), tracker
            )

        with tracker.stage("asr"):
            transcript = self.asr.transcribe(trimmed)

        # Nothing was actually said. Short-circuit rather than spend a remote
        # round trip asking a model what "..." means.
        if not _HAS_CONTENT.search(transcript.text or ""):
            return self._nothing_heard(transcript, tracker)

        return self._finish(transcript, transcript.text.strip(), conversation, tracker)

    def respond_to_text(self, text: str, conversation: Conversation) -> TurnResult:
        """Same turn, skipping the microphone.

        Used by tests and by the API's text endpoint. Going through the same
        path keeps one implementation of the turn rather than two that drift.
        """
        tracker = LatencyTracker()
        transcript = Transcript(text=text, confidence=1.0, duration_s=0.0)

        if not _HAS_CONTENT.search(text or ""):
            return self._nothing_heard(transcript, tracker)

        return self._finish(transcript, text.strip(), conversation, tracker)

    # --- internals -----------------------------------------------------------

    def _finish(
        self,
        transcript: Transcript,
        text: str,
        conversation: Conversation,
        tracker: LatencyTracker,
    ) -> TurnResult:
        with tracker.stage("agent"):
            result = self.agent.handle(text, conversation)

        with tracker.stage("tts"):
            audio = self._speak(result.text)

        turn = TurnResult(
            transcript=transcript,
            text=text,
            agent=result,
            audio=audio,
            latency=tracker.breakdown(),
        )
        self._log(turn)
        return turn

    def resolve_pending(self, confirmed: bool, conversation: Conversation) -> TurnResult:
        """The Yes or Cancel button: act, then let the model say what happened."""
        tracker = LatencyTracker()
        transcript = Transcript(text="yes" if confirmed else "no", confidence=1.0, duration_s=0.0)

        with tracker.stage("agent"):
            result = self.agent.resolve_pending(confirmed, conversation)

        with tracker.stage("tts"):
            audio = self._speak(result.text)

        turn = TurnResult(
            transcript=transcript,
            text=transcript.text,
            agent=result,
            audio=audio,
            latency=tracker.breakdown(),
        )
        self._log(turn)
        return turn

    def _speak(self, text: str) -> AudioBuffer:
        if not self.speak:
            return AudioBuffer(samples=SILENCE, sample_rate=16_000)
        return self.tts.synthesize(text)

    def _nothing_heard(self, transcript: Transcript, tracker: LatencyTracker) -> TurnResult:
        """Silence, or a transcript with nothing in it. The model is not called."""
        with tracker.stage("agent"):
            pass  # named so every turn reports the same four stages
        with tracker.stage("tts"):
            audio = self._speak(NOT_CAUGHT)
        return TurnResult(
            transcript=transcript,
            text="",
            agent=AgentResult(text=NOT_CAUGHT),
            audio=audio,
            latency=tracker.breakdown(),
        )

    def _log(self, result: TurnResult) -> None:
        slowest = result.latency.slowest()
        log.info(
            "heard=%r tools=%s guard_hit=%s total=%.0fms slowest=%s",
            result.text,
            [call.name for call in result.agent.tool_calls] or "none",
            result.agent.guard_hit,
            result.latency.total_ms,
            f"{slowest[0]} {slowest[1]:.0f}ms" if slowest else "n/a",
        )
