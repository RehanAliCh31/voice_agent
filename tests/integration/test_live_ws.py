"""The live session through its WebSocket, /v1/live, end to end on fake engines."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.live_support import LevelVad, PiecesVoice, ScriptedASR, quiet, tone
from voicebot.api.app import create_app
from voicebot.llm.base import FakeLLMClient, Reply

CHUNK = 320


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app()) as test_client:
        services = test_client.app.state.services  # type: ignore[attr-defined]
        services.tts = PiecesVoice()
        services.live_vad_factory = LevelVad
        yield test_client


def _script(client: TestClient, heard: list[tuple[str, str]], replies: list[Reply]) -> Any:
    services = client.app.state.services  # type: ignore[attr-defined]
    services.asr = ScriptedASR(heard)
    services.llm = FakeLLMClient(replies)
    return services


def _stream(ws: Any, *parts: np.ndarray) -> None:
    audio = np.concatenate(parts)
    for start in range(0, len(audio), CHUNK):
        ws.send_bytes(audio[start : start + CHUNK].astype("<i2").tobytes())


def _until(ws: Any, kind: str, limit: int = 400) -> list[dict[str, Any] | bytes]:
    """Every message up to and including the first of `kind`."""
    seen: list[dict[str, Any] | bytes] = []
    for _ in range(limit):
        message = ws.receive()
        if message.get("bytes") is not None:
            seen.append(message["bytes"])
            continue
        if message.get("text") is None:
            break
        event = json.loads(message["text"])
        seen.append(event)
        if event["type"] == kind:
            return seen
    raise AssertionError(f"no {kind}: got {[e if isinstance(e, dict) else 'audio' for e in seen]}")


def _events(seen: list[dict[str, Any] | bytes], kind: str) -> list[dict[str, Any]]:
    return [e for e in seen if isinstance(e, dict) and e["type"] == kind]


def test_a_conversation_by_voice_alone(client: TestClient) -> None:
    _script(
        client,
        [("what is my balance", "en"), ("thank you", "en")],
        [Reply(text="Which account. Checking or savings?"), Reply(text="You're welcome.")],
    )
    with client.websocket_connect("/v1/live") as ws:
        ws.send_text(json.dumps({"type": "start"}))
        started = _until(ws, "started")
        session_id = _events(started, "started")[0]["session_id"]
        _stream(ws, tone(900), quiet(700))
        seen = _until(ws, "audio_end")
        assert _events(seen, "transcript")[0]["text"] == "what is my balance"
        reply = _events(seen, "reply")[0]
        assert reply["reply"] == "Which account. Checking or savings?"
        assert "audio_b64" not in reply  # the audio comes as its own messages
        # Each piece's header, then its WAV.
        assert [e["piece"] for e in _events(seen, "audio")] == [0, 1]
        assert sum(isinstance(e, bytes) and e[:4] == b"RIFF" for e in seen) == 2
        ws.send_text(json.dumps({"type": "playback_done", "audio_id": reply["audio_id"]}))
        assert _events(_until(ws, "state"), "state")[-1]["state"] == "listening"
        _stream(ws, tone(900), quiet(700))
        assert _events(_until(ws, "reply"), "reply")[0]["reply"] == "You're welcome."
        ws.send_text(json.dumps({"type": "stop"}))
        assert _events(_until(ws, "ended"), "ended")[0]["reason"] == "the caller stopped"
    # The conversation is the page's: the same history as a typed one.
    services = client.app.state.services  # type: ignore[attr-defined]
    users = [
        m["content"]
        for m in services.sessions.get(session_id).conversation.messages
        if m["role"] == "user"
    ]
    assert len(users) == 2


def test_speaking_over_the_bot_through_the_socket(client: TestClient) -> None:
    _script(
        client,
        [("tell me everything", "en"), ("stop, just savings", "en")],
        [Reply(text="Here is a long answer. With many parts. And more."), Reply(text="Savings.")],
    )
    with client.websocket_connect("/v1/live") as ws:
        ws.send_text(json.dumps({"type": "start"}))
        _until(ws, "started")
        _stream(ws, tone(900), quiet(700))
        reply = _events(_until(ws, "reply"), "reply")[0]
        _stream(ws, tone(1000), quiet(700))
        seen = _until(ws, "stop_audio")
        ws.send_text(
            json.dumps(
                {"type": "interrupted", "audio_id": reply["audio_id"], "piece": 0, "offset_ms": 300}
            )
        )
        seen = _until(ws, "reply")
        assert _events(seen, "reply")[0]["reply"] == "Savings."
        ws.send_text(json.dumps({"type": "stop"}))
        _until(ws, "ended")


def test_a_second_start_on_one_connection_is_ignored(client: TestClient) -> None:
    _script(client, [], [])
    with client.websocket_connect("/v1/live") as ws:
        ws.send_text(json.dumps({"type": "start"}))
        ws.send_text(json.dumps({"type": "start"}))
        ws.send_text("not json")
        ws.send_text(json.dumps({"type": "ping"}))
        seen = _until(ws, "pong")
        assert len(_events(seen, "started")) == 1
        ws.send_text(json.dumps({"type": "stop"}))
        _until(ws, "ended")


def test_a_caller_who_drops_off_mid_turn_leaves_nothing_running(client: TestClient) -> None:
    services = _script(client, [("hello", "en")], [Reply(text="Hi.")])
    services.asr.delay_s = 0.5  # still recognising when the caller goes
    with client.websocket_connect("/v1/live") as ws:
        ws.send_text(json.dumps({"type": "start"}))
        _until(ws, "started")
        _stream(ws, tone(900), quiet(700))
        _until(ws, "state")  # listening -> user_speaking ...
    # The socket closed without a stop: the session is stopped all the same,
    # and its threads -- audio, turn, voice -- all finish.
    finish = time.monotonic() + 5
    while _live_threads() and time.monotonic() < finish:
        time.sleep(0.05)
    assert _live_threads() == []


def _live_threads() -> list[str]:
    return [t.name for t in threading.enumerate() if t.name.startswith("live-")]


def test_a_dropped_caller_carries_on_the_same_conversation(client: TestClient) -> None:
    services = _script(
        client,
        [("my name is Ali", "en"), ("what did I say", "en")],
        [Reply(text="Hello Ali."), Reply(text="You said your name.")],
    )
    with client.websocket_connect("/v1/live") as ws:
        ws.send_text(json.dumps({"type": "start"}))
        session_id = _events(_until(ws, "started"), "started")[0]["session_id"]
        _stream(ws, tone(900), quiet(700))
        _until(ws, "audio_end")
    with client.websocket_connect("/v1/live") as ws:  # the page reconnects
        ws.send_text(json.dumps({"type": "start", "session_id": session_id}))
        assert _events(_until(ws, "started"), "started")[0]["session_id"] == session_id
        _stream(ws, tone(900), quiet(700))
        _until(ws, "reply")
        ws.send_text(json.dumps({"type": "stop"}))
        _until(ws, "ended")
    users = [
        m for m in services.sessions.get(session_id).conversation.messages if m["role"] == "user"
    ]
    assert len(users) == 2


def test_no_vad_model_says_so_and_ends(client: TestClient) -> None:
    services = client.app.state.services  # type: ignore[attr-defined]

    def missing() -> Any:
        raise FileNotFoundError("no silero_vad.onnx")

    services.live_vad_factory = missing
    with client.websocket_connect("/v1/live") as ws:
        ws.send_text(json.dumps({"type": "start"}))
        seen = _until(ws, "ended")
        assert _events(seen, "error")[0]["message"]
