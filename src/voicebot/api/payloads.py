"""Request and response bodies for the HTTP API.

Separate from `voicebot.schemas` on purpose. Those dataclasses are the
pipeline's internal contract; these are the *published* one. Serving the
internals directly would mean any refactor is a breaking API change, and the
frontend would have to follow every rename.
"""

from __future__ import annotations

import base64

from pydantic import BaseModel, Field

from voicebot.pipeline.audio_io import encode_wav
from voicebot.pipeline.orchestrator import TurnResult
from voicebot.schemas import Understanding


class SlotPayload(BaseModel):
    """One extracted parameter, with its span in the utterance."""

    name: str
    value: str
    start: int
    end: int
    confidence: float


class TranscriptPayload(BaseModel):
    text: str
    confidence: float
    duration_s: float


class ActionPayload(BaseModel):
    """What the bot did, and why. The frontend renders by `type`."""

    type: str
    text: str
    intent: str | None = None
    filled_slots: dict[str, str] = Field(default_factory=dict)
    missing_slots: list[str] = Field(default_factory=list)


class UnderstandingPayload(BaseModel):
    intent: str
    intent_confidence: float
    slots: list[SlotPayload] = Field(default_factory=list)
    oos_score: float
    is_out_of_scope: bool


class TurnPayload(BaseModel):
    """Everything one turn produced, including what the inspector panel shows."""

    session_id: str
    transcript: TranscriptPayload
    text: str
    understanding: UnderstandingPayload
    action: ActionPayload
    latency_ms: dict[str, float]
    audio_b64: str | None = None
    audio_sample_rate: int = 16_000


class TextRequest(BaseModel):
    """Type instead of speaking. Used by the frontend's fallback input."""

    text: str
    session_id: str | None = None


class SpeakRequest(BaseModel):
    text: str


class AudioPayload(BaseModel):
    audio_b64: str
    sample_rate: int
    duration_s: float


class HealthPayload(BaseModel):
    status: str
    version: str


class ReadyPayload(BaseModel):
    ready: bool
    asr: str
    nlu: str
    tts: str
    device: str
    sessions: int


def understanding_payload(understanding: Understanding) -> UnderstandingPayload:
    return UnderstandingPayload(
        intent=understanding.intent,
        intent_confidence=understanding.intent_confidence,
        slots=[
            SlotPayload(
                name=slot.name,
                value=slot.value,
                start=slot.start,
                end=slot.end,
                confidence=slot.confidence,
            )
            for slot in understanding.slots
        ],
        oos_score=understanding.oos_score,
        is_out_of_scope=understanding.is_out_of_scope,
    )


def turn_payload(session_id: str, result: TurnResult, include_audio: bool = True) -> TurnPayload:
    """Convert a pipeline result into the published shape.

    Audio is base64-encoded WAV inside the JSON rather than a second request
    for a binary body. One round trip per turn keeps the client simple, and a
    reply is a few hundred kilobytes.
    """
    audio_b64 = None
    if include_audio and len(result.audio.samples):
        audio_b64 = base64.b64encode(encode_wav(result.audio)).decode("ascii")

    return TurnPayload(
        session_id=session_id,
        transcript=TranscriptPayload(
            text=result.transcript.text,
            confidence=result.transcript.confidence,
            duration_s=result.transcript.duration_s,
        ),
        text=result.text,
        understanding=understanding_payload(result.understanding),
        action=ActionPayload(
            type=result.action.type.value,
            text=result.action.text,
            intent=result.action.intent,
            filled_slots=dict(result.action.filled_slots),
            missing_slots=list(result.action.missing_slots),
        ),
        latency_ms={key: round(value, 2) for key, value in result.latency.as_dict().items()},
        audio_b64=audio_b64,
        audio_sample_rate=result.audio.sample_rate,
    )
