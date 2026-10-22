"""The HTTP surface.

A whole turn, the two confirmation buttons, a reset, and the operational
endpoints. The per-stage debugging routes from v1 are gone with the stages
they exposed; what the inspector panel shows now is the tool calls, which
come back with the turn itself.
"""

from __future__ import annotations

import base64

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

from voicebot import __version__
from voicebot.api.deps import Services, get_services
from voicebot.api.payloads import (
    AudioPayload,
    HealthPayload,
    ReadyPayload,
    SessionRequest,
    SpeakRequest,
    TextRequest,
    TranscriptPayload,
    TurnPayload,
    turn_payload,
)
from voicebot.api.sessions import SessionStore
from voicebot.pipeline.audio_io import decode_wav, encode_wav
from voicebot.pipeline.orchestrator import TurnResult
from voicebot.schemas import AudioBuffer
from voicebot.utils.logging import get_logger

log = get_logger("api.routes")

router = APIRouter()

MAX_AUDIO_BYTES = 10 * 1024 * 1024  # ~5 minutes of 16 kHz mono


def _read_audio(data: bytes) -> AudioBuffer:
    """Decode an uploaded WAV, refusing anything unusable with a clear reason."""
    if not data:
        raise HTTPException(status_code=400, detail="No audio was uploaded.")
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Audio is {len(data) // 1024} kB; the limit is {MAX_AUDIO_BYTES // 1024} kB.",
        )
    try:
        return decode_wav(data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read the audio: {exc}") from exc


def _session_id(store: SessionStore, given: str | None) -> str:
    return given or store.new_id()


def _publish(services: Services, session_id: str, result: TurnResult) -> TurnPayload:
    """Count it, log it, and convert it into the published shape."""
    session = services.sessions.get(session_id)
    services.record_turn(session_id, session.conversation.turn_index, result)
    return turn_payload(
        session_id,
        result,
        services.speaks_client_side,
        awaiting_confirmation=session.executor.pending is not None,
    )


# --- operational -------------------------------------------------------------


@router.get("/health", response_model=HealthPayload, tags=["ops"])
def health() -> HealthPayload:
    """Liveness. Answers before the models have loaded."""
    return HealthPayload(status="ok", version=__version__)


@router.get("/ready", response_model=ReadyPayload, tags=["ops"])
def ready(services: Services = Depends(get_services)) -> ReadyPayload:
    """Readiness. Names the engines actually loaded, not the ones configured."""
    described = services.describe()
    return ReadyPayload(
        ready=True,
        asr=described["asr"],
        llm=described["llm"],
        tts=described["tts"],
        device=services.device,
        sessions=len(services.sessions),
    )


@router.get("/metrics", response_class=PlainTextResponse, tags=["ops"])
def metrics(services: Services = Depends(get_services)) -> str:
    """Prometheus text exposition, hand-rolled.

    The guard hit rate is the number worth watching. A rising one means the
    model is reaching for figures it was not given, which is the failure this
    whole design exists to prevent.
    """
    lines = [
        "# HELP voicebot_turns_total Conversational turns served.",
        "# TYPE voicebot_turns_total counter",
        f"voicebot_turns_total {services.turns}",
        "# HELP voicebot_tool_calls_total Tool invocations by name.",
        "# TYPE voicebot_tool_calls_total counter",
    ]
    for tool, count in sorted(services.tool_calls.items()):
        lines.append(f'voicebot_tool_calls_total{{tool="{tool}"}} {count}')
    lines += [
        "# HELP voicebot_guard_hits_total Replies suppressed for stating an unsourced figure.",
        "# TYPE voicebot_guard_hits_total counter",
        f"voicebot_guard_hits_total {services.bank.guard_hit_count()}",
        "# HELP voicebot_sessions Active conversations.",
        "# TYPE voicebot_sessions gauge",
        f"voicebot_sessions {len(services.sessions)}",
    ]
    return "\n".join(lines) + "\n"


# --- single stages, for debugging -------------------------------------------


@router.post("/v1/transcribe", response_model=TranscriptPayload, tags=["stages"])
async def transcribe(
    audio: UploadFile = File(..., description="16-bit PCM WAV"),
    services: Services = Depends(get_services),
) -> TranscriptPayload:
    """Audio to text. The ASR stage on its own."""
    buffer = _read_audio(await audio.read())
    transcript = services.asr.transcribe(buffer)
    return TranscriptPayload(
        text=transcript.text,
        confidence=transcript.confidence,
        duration_s=transcript.duration_s,
    )


@router.post("/v1/speak", response_model=AudioPayload, tags=["stages"])
def speak(request: SpeakRequest, services: Services = Depends(get_services)) -> AudioPayload:
    """Text to audio. The TTS stage on its own.

    Returns empty audio when the page is doing the speaking, which is the
    honest answer rather than an error.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Nothing to say.")
    buffer = services.tts.synthesize(request.text)
    return AudioPayload(
        audio_b64=base64.b64encode(encode_wav(buffer)).decode("ascii"),
        sample_rate=buffer.sample_rate,
        duration_s=buffer.duration_s,
    )


# --- a whole turn ------------------------------------------------------------


@router.post("/v1/converse", response_model=TurnPayload, tags=["conversation"])
async def converse(
    audio: UploadFile = File(..., description="16-bit PCM WAV"),
    session_id: str | None = Form(default=None),
    services: Services = Depends(get_services),
) -> TurnPayload:
    """Audio in, audio out. The whole pipeline, with conversation state."""
    buffer = _read_audio(await audio.read())
    sid = _session_id(services.sessions, session_id)
    session = services.sessions.get(sid)

    result = services.pipeline(session).converse(buffer, session.conversation)

    services.sessions.put(sid, session)
    return _publish(services, sid, result)


@router.post("/v1/chat", response_model=TurnPayload, tags=["conversation"])
def chat(request: TextRequest, services: Services = Depends(get_services)) -> TurnPayload:
    """Type instead of speaking, same turn otherwise."""
    sid = _session_id(services.sessions, request.session_id)
    session = services.sessions.get(sid)

    result = services.pipeline(session).respond_to_text(request.text, session.conversation)

    services.sessions.put(sid, session)
    return _publish(services, sid, result)


# --- confirmation ------------------------------------------------------------


@router.post("/v1/confirm", response_model=TurnPayload, tags=["conversation"])
def confirm(request: SessionRequest, services: Services = Depends(get_services)) -> TurnPayload:
    """Agree to the pending action.

    This acts on the executor directly, without asking the model whether it
    agrees -- which is what makes a misheard "yes" harmless. The model is then
    told what happened, so its next turn knows the money moved.
    """
    return _resolve(request, services, confirmed=True)


@router.post("/v1/cancel", response_model=TurnPayload, tags=["conversation"])
def cancel(request: SessionRequest, services: Services = Depends(get_services)) -> TurnPayload:
    """Decline the pending action."""
    return _resolve(request, services, confirmed=False)


def _resolve(request: SessionRequest, services: Services, confirmed: bool) -> TurnPayload:
    sid = _session_id(services.sessions, request.session_id)
    session = services.sessions.get(sid)

    result = services.pipeline(session).resolve_pending(confirmed, session.conversation)

    services.sessions.put(sid, session)
    return _publish(services, sid, result)


# --- housekeeping ------------------------------------------------------------


@router.post("/v1/reset", tags=["conversation"])
def reset(request: SessionRequest, services: Services = Depends(get_services)) -> dict[str, str]:
    """Put the demo bank back to its opening balances.

    v1 got this for free by throwing away a per-session bank. With one shared
    database, repeated demos would otherwise drain checking until every
    transfer failed for want of funds. The audit log survives.
    """
    services.bank.reset()
    if request.session_id:
        services.sessions.drop(request.session_id)
    return {"status": "reset"}


@router.delete("/v1/session/{session_id}", tags=["conversation"])
def end_session(session_id: str, services: Services = Depends(get_services)) -> dict[str, str]:
    """Forget a conversation, leaving the money alone."""
    services.sessions.drop(session_id)
    return {"status": "ended", "session_id": session_id}
