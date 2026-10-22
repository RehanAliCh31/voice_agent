"""The HTTP surface.

Four working endpoints and three operational ones. Each pipeline stage is
exposed on its own so the frontend can debug a turn piece by piece, and
`/v1/converse` runs the lot.
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
    SpeakRequest,
    TextRequest,
    TranscriptPayload,
    TurnPayload,
    UnderstandingPayload,
    turn_payload,
    understanding_payload,
)
from voicebot.api.sessions import SessionStore
from voicebot.pipeline.audio_io import decode_wav, encode_wav
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
        nlu=described["nlu"],
        tts=described["tts"],
        device=services.device,
        sessions=len(services.sessions),
    )


@router.get("/metrics", response_class=PlainTextResponse, tags=["ops"])
def metrics(services: Services = Depends(get_services)) -> str:
    """Prometheus text exposition, hand-rolled.

    A counter per action type is enough to see what the bot is actually doing
    in production -- a rising `out_of_scope` or `clarify` rate is the first
    sign the ASR or the model has drifted. Not worth a dependency.
    """
    lines = [
        "# HELP voicebot_turns_total Conversational turns served.",
        "# TYPE voicebot_turns_total counter",
        f"voicebot_turns_total {services.turns}",
        "# HELP voicebot_actions_total Turns by action type.",
        "# TYPE voicebot_actions_total counter",
    ]
    for action, count in sorted(services.action_counts.items()):
        lines.append(f'voicebot_actions_total{{action="{action}"}} {count}')
    lines += [
        "# HELP voicebot_sessions Active conversations.",
        "# TYPE voicebot_sessions gauge",
        f"voicebot_sessions {len(services.sessions)}",
    ]
    return "\n".join(lines) + "\n"


# --- one stage at a time -----------------------------------------------------


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


@router.post("/v1/understand", response_model=UnderstandingPayload, tags=["stages"])
def understand(
    request: TextRequest, services: Services = Depends(get_services)
) -> UnderstandingPayload:
    """Text to intent and slots. The NLU stage on its own."""
    from voicebot.asr.normalizer import normalize

    return understanding_payload(services.nlu.understand(normalize(request.text)))


@router.post("/v1/speak", response_model=AudioPayload, tags=["stages"])
def speak(request: SpeakRequest, services: Services = Depends(get_services)) -> AudioPayload:
    """Text to audio. The TTS stage on its own."""
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
    conversation = services.sessions.get(sid)

    result = services.pipeline(conversation.bank).converse(buffer, conversation.state)

    services.sessions.put(sid, conversation)
    services.record(result.action.type.value)
    return turn_payload(sid, result)


@router.post("/v1/chat", response_model=TurnPayload, tags=["conversation"])
def chat(request: TextRequest, services: Services = Depends(get_services)) -> TurnPayload:
    """Type instead of speaking, same turn otherwise.

    The frontend uses this when the microphone is unavailable, and it makes
    the whole conversation testable without synthesising audio.
    """
    sid = _session_id(services.sessions, request.session_id)
    conversation = services.sessions.get(sid)

    result = services.pipeline(conversation.bank).respond_to_text(request.text, conversation.state)

    services.sessions.put(sid, conversation)
    services.record(result.action.type.value)
    return turn_payload(sid, result)


@router.delete("/v1/session/{session_id}", tags=["conversation"])
def end_session(session_id: str, services: Services = Depends(get_services)) -> dict[str, str]:
    """Forget a conversation. The frontend calls this on reset."""
    services.sessions.drop(session_id)
    return {"status": "ended", "session_id": session_id}
