"""The live session's WebSocket, /v1/live.

The page connects, says `start`, and streams the microphone as binary
messages: 16 kHz, 16-bit, little-endian, mono PCM, in chunks of any length
(it sends 20 ms). It may also send, as JSON:

    {"type": "start", "session_id": "..."}   begin (or carry on a conversation)
    {"type": "stop"}                          end it
    {"type": "interrupted", "audio_id": "...", "piece": 2, "offset_ms": 840}
                                              where playback stopped, when told to stop
    {"type": "playback_done", "audio_id": "..."}
                                              a reply has been heard to its end
    {"type": "ping"}

The service sends JSON events -- `started`, `state`, `transcript`, `reply`
and `say` (the turn, as the push-to-talk page gets it, less its audio),
`audio` (a piece's header, followed at once by the piece itself as a binary
WAV message), `audio_end`, `audio_failed`, `stop_audio` (barge-in: stop
now, keep what you have), `resume_audio` (it was only a "mm-hmm"),
`discard_audio`, `metrics`, `error`, `ended`, `pong` -- in the order they
happened, on one connection.

Whatever ends the connection -- stop, the caller leaving, an error -- the
session is stopped and its threads waited for, in a `finally`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool

from voicebot.errors import InternalError
from voicebot.live.session import LiveSession
from voicebot.messages import say
from voicebot.utils.logging import get_logger
from voicebot.utils.observability import count, report

log = get_logger("live.routes")

router = APIRouter()

# A connection whose page falls this far behind on what it is sent is not
# reading: it is let go, rather than held in memory.
OUTBOX_LIMIT = 2000


@router.websocket("/v1/live")
async def live(websocket: WebSocket) -> None:
    await websocket.accept()
    services = websocket.app.state.services
    loop = asyncio.get_running_loop()
    outbox: asyncio.Queue[dict[str, Any] | bytes | None] = asyncio.Queue()
    open_ = True

    def send(message: dict[str, Any] | bytes) -> None:
        """From any thread: queued for the page, in order."""
        if open_:
            loop.call_soon_threadsafe(_put, outbox, message)

    sender = asyncio.create_task(_drain(websocket, outbox))
    session: LiveSession | None = None
    count("live_sessions")
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            data = message.get("bytes")
            if data is not None:
                if session is not None:
                    session.feed(data)
                continue
            control = _control(message.get("text"))
            if control is None:
                continue
            kind = control.get("type")
            if kind == "start" and session is None:
                session = await _start(services, control, send)
                if session is None:
                    break
            elif kind == "stop":
                if session is not None:
                    session.stop("the caller stopped")
                break
            elif kind == "interrupted" and session is not None:
                session.played(
                    str(control.get("audio_id", "")),
                    int(control.get("piece", 0)),
                    float(control.get("offset_ms", 0.0)),
                )
            elif kind == "playback_done" and session is not None:
                session.playback_done(str(control.get("audio_id", "")))
            elif kind == "ping":
                send({"type": "pong"})
            if session is not None and session.state.value == "ended":
                break
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # anything at all: the session still ends cleanly below
        log.error("the live connection failed: %s", type(exc).__name__, exc_info=exc)
        report(InternalError("the live connection failed", cause=exc))
    finally:
        if session is not None:
            await run_in_threadpool(session.close)
        # After whatever is already queued -- an error, "ended" -- which the
        # page must still be sent.
        loop.call_soon(_put, outbox, None)
        open_ = False
        with contextlib.suppress(Exception):
            await asyncio.wait_for(sender, timeout=2.0)
        with contextlib.suppress(Exception):
            await websocket.close()


async def _start(services: Any, control: dict[str, Any], send: Any) -> LiveSession | None:
    session_id = str(control.get("session_id") or "") or services.sessions.new_id()
    try:
        vad = await run_in_threadpool(services.make_live_vad)
    except Exception as exc:  # no VAD model: the live session cannot hear anything
        log.error("no VAD for the live session: %s", exc)
        report(InternalError("the live session has no VAD", cause=exc))
        language = services.sessions.get(session_id).conversation.language
        send({"type": "error", "message": say("speech_unavailable", language)})
        send({"type": "ended", "reason": "no voice detection"})
        return None
    session = LiveSession(services, session_id, send, vad)
    live = services.config.live
    send(
        {
            "type": "started",
            "session_id": session_id,
            "end_silence_ms": live.end_silence_ms,
            "sample_rate": 16000,
        }
    )
    await run_in_threadpool(session.start)
    return session


def _control(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    try:
        value = json.loads(text)
    except ValueError:
        log.info("ignored a live message that was not JSON")
        return None
    return value if isinstance(value, dict) else None


def _put(outbox: asyncio.Queue[dict[str, Any] | bytes | None], message: Any) -> None:
    if outbox.qsize() > OUTBOX_LIMIT and message is not None:
        return
    outbox.put_nowait(message)


async def _drain(
    websocket: WebSocket, outbox: asyncio.Queue[dict[str, Any] | bytes | None]
) -> None:
    """Send what the session says, in order, until it is over."""
    while True:
        message = await outbox.get()
        if message is None:
            return
        try:
            if isinstance(message, bytes):
                await websocket.send_bytes(message)
            else:
                await websocket.send_text(json.dumps(message, ensure_ascii=False))
        except Exception:  # the page has gone: nothing more can reach it
            return
        if isinstance(message, dict) and message.get("type") == "ended":
            with contextlib.suppress(Exception):
                await websocket.close(code=1000)
            return
