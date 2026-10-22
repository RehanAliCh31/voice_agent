"""Request and response bodies for the HTTP API.

Separate from `voicebot.schemas` on purpose. Those dataclasses are the
pipeline's internal contract; these are the *published* one. Serving the
internals directly would mean any refactor is a breaking API change, and the
front end would have to follow every rename.
"""

from __future__ import annotations

import base64
from typing import Any

from pydantic import BaseModel, Field

from voicebot.pipeline.audio_io import encode_wav
from voicebot.pipeline.orchestrator import TurnResult


class TranscriptPayload(BaseModel):
    text: str
    confidence: float
    duration_s: float


class ToolCallPayload(BaseModel):
    """One tool the model ran, with what it asked for and what came back.

    This is what the inspector panel shows in place of v1's intent, slots and
    gate score: the actual work done, rather than a classifier's opinion.
    """

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)


class TurnPayload(BaseModel):
    """Everything one turn produced, including what the inspector panel shows."""

    session_id: str
    transcript: TranscriptPayload
    text: str
    reply: str
    tool_calls: list[ToolCallPayload] = Field(default_factory=list)
    guard_hit: bool = False
    # True while an action has been read back and not yet agreed to. The page
    # shows its Yes/Cancel buttons from this.
    awaiting_confirmation: bool = False
    latency_ms: dict[str, float] = Field(default_factory=dict)
    # Null when the page is doing the speaking. Nullable rather than absent so
    # a client can tell "no audio" from "field missing".
    audio_b64: str | None = None
    speak_client_side: bool = False
    audio_sample_rate: int = 16_000


class TextRequest(BaseModel):
    """Type instead of speaking. Used by the front end's fallback input."""

    text: str
    session_id: str | None = None


class SessionRequest(BaseModel):
    """Act on a session without saying anything -- confirm, cancel, reset."""

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
    llm: str
    tts: str
    device: str
    sessions: int


def turn_payload(
    session_id: str,
    result: TurnResult,
    speaks_client_side: bool,
    awaiting_confirmation: bool = False,
) -> TurnPayload:
    """Convert a pipeline result into the published shape.

    Audio is base64-encoded WAV inside the JSON rather than a second request
    for a binary body. One round trip per turn keeps the client simple, and a
    reply is a few hundred kilobytes. With browser speech there is no audio at
    all, and `speak_client_side` tells the page to say the words itself.
    """
    audio_b64 = None
    if len(result.audio.samples):
        audio_b64 = base64.b64encode(encode_wav(result.audio)).decode("ascii")

    # The token gates money. There is no reason to publish it, even
    # same-origin to the session that owns it.
    calls = [
        ToolCallPayload(
            name=call.name,
            arguments={k: v for k, v in call.arguments.items() if k != "confirmation_token"},
            result={k: v for k, v in outcome.items() if k != "token"},
        )
        for call, outcome in zip(result.agent.tool_calls, result.agent.tool_results, strict=False)
    ]

    return TurnPayload(
        session_id=session_id,
        transcript=TranscriptPayload(
            text=result.transcript.text,
            confidence=result.transcript.confidence,
            duration_s=result.transcript.duration_s,
        ),
        text=result.text,
        reply=result.agent.text,
        tool_calls=calls,
        guard_hit=result.agent.guard_hit,
        awaiting_confirmation=awaiting_confirmation,
        latency_ms={key: round(value, 2) for key, value in result.latency.as_dict().items()},
        audio_b64=audio_b64,
        speak_client_side=speaks_client_side,
        audio_sample_rate=result.audio.sample_rate,
    )
