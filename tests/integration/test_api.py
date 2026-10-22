"""The HTTP surface, driven through FastAPI's TestClient.

No VOICEBOT_CONFIG is set, so the app builds with the fake engines: the whole
API is exercised with no checkpoint, no download and no GPU.
"""

import base64

import numpy as np
import pytest
from fastapi.testclient import TestClient

from voicebot.api.app import create_app
from voicebot.pipeline.audio_io import decode_wav, encode_wav
from voicebot.schemas import AudioBuffer


def _wav(seconds: float = 1.0, rate: int = 16_000, silent: bool = False) -> bytes:
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False, dtype=np.float32)
    samples = np.zeros_like(t) if silent else (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return encode_wav(AudioBuffer(samples=samples.astype(np.float32), sample_rate=rate))


@pytest.fixture
def client():
    # The context manager is what runs the lifespan, which loads the engines.
    with TestClient(create_app()) as test_client:
        yield test_client


# --- operational -------------------------------------------------------------


def test_health_reports_ok_and_a_version(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["version"]


def test_ready_names_the_engines_actually_loaded(client):
    body = client.get("/ready").json()
    assert body["ready"] is True
    assert body["asr"] == "FakeASREngine"
    assert body["nlu"] == "FakeNLUEngine"
    assert body["tts"] == "FakeTTSEngine"
    assert body["device"] in {"cpu", "cuda"}


def test_metrics_is_prometheus_text(client):
    client.post("/v1/chat", json={"text": "what is my balance"})
    text = client.get("/metrics").text
    assert "voicebot_turns_total" in text
    assert "# TYPE voicebot_turns_total counter" in text
    assert "voicebot_actions_total{action=" in text


def test_metrics_counts_turns(client):
    before = client.get("/metrics").text
    assert "voicebot_turns_total 0" in before
    client.post("/v1/chat", json={"text": "what is my balance"})
    client.post("/v1/chat", json={"text": "what is my balance"})
    assert "voicebot_turns_total 2" in client.get("/metrics").text


# --- one stage at a time -----------------------------------------------------


def test_transcribe_returns_text_and_confidence(client):
    response = client.post("/v1/transcribe", files={"audio": ("a.wav", _wav(), "audio/wav")})
    assert response.status_code == 200
    body = response.json()
    assert body["text"]  # the fake returns a fixed sentence
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["duration_s"] == pytest.approx(1.0, abs=0.01)


def test_understand_returns_intent_and_scores(client):
    body = client.post("/v1/understand", json={"text": "What is my balance?"}).json()
    assert body["intent"]
    assert 0.0 <= body["intent_confidence"] <= 1.0
    assert "is_out_of_scope" in body


def test_understand_normalises_before_classifying(client):
    """The endpoint must apply the same normalisation the pipeline does."""
    body = client.post("/v1/understand", json={"text": "   "}).json()
    assert body["is_out_of_scope"] is True


def test_speak_returns_decodable_wav(client):
    body = client.post("/v1/speak", json={"text": "your balance is fifty dollars"}).json()
    audio = decode_wav(base64.b64decode(body["audio_b64"]))
    assert audio.sample_rate == body["sample_rate"]
    assert audio.duration_s > 0


def test_speak_rejects_empty_text(client):
    assert client.post("/v1/speak", json={"text": "   "}).status_code == 400


# --- a whole turn ------------------------------------------------------------


def test_chat_returns_the_full_inspector_payload(client):
    body = client.post("/v1/chat", json={"text": "what is my checking balance"}).json()

    assert body["session_id"]
    assert body["text"] == "what is my checking balance"
    assert "intent" in body["understanding"]
    assert body["action"]["type"]
    assert body["action"]["text"]
    assert "total" in body["latency_ms"]
    assert body["audio_b64"]


def test_converse_accepts_audio_and_returns_audio(client):
    response = client.post(
        "/v1/converse",
        files={"audio": ("a.wav", _wav(), "audio/wav")},
        data={"session_id": "fixed-session"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == "fixed-session"
    assert body["transcript"]["text"]
    assert decode_wav(base64.b64decode(body["audio_b64"])).duration_s > 0


def test_silence_is_answered_with_a_clarification(client):
    body = client.post(
        "/v1/converse", files={"audio": ("a.wav", _wav(silent=True), "audio/wav")}
    ).json()
    assert body["action"]["type"] == "clarify"
    assert body["text"] == ""


# --- sessions ----------------------------------------------------------------


def test_a_session_id_is_issued_when_none_is_given(client):
    first = client.post("/v1/chat", json={"text": "hello"}).json()
    second = client.post("/v1/chat", json={"text": "hello"}).json()
    assert first["session_id"] != second["session_id"]


def test_conversation_state_persists_within_a_session(client):
    first = client.post("/v1/chat", json={"text": "transfer money"}).json()
    sid = first["session_id"]
    assert first["action"]["type"] == "ask_slot"

    second = client.post("/v1/chat", json={"text": "transfer money", "session_id": sid}).json()
    assert second["session_id"] == sid
    # The second turn continues the same task rather than starting over.
    assert second["action"]["intent"] == first["action"]["intent"]


def test_a_transfer_persists_across_turns_in_one_session(client):
    """The bug this pins.

    The bank used to be rebuilt per request, so a transfer completed, the
    reply said the balance had changed, and the next turn read the original
    balance again. The money never actually moved.
    """
    sid = "money-session"
    services = client.app.state.services

    services.sessions.get(sid).bank.transfer(5_000, "sarah", "checking")

    # A LATER request in the same session must see the reduced balance.
    client.post("/v1/chat", json={"text": "what is my balance", "session_id": sid})
    assert services.sessions.get(sid).bank.balances["checking"] == 243_350


def test_two_sessions_have_independent_accounts(client):
    services = client.app.state.services
    a = services.sessions.get("session-a").bank
    b = services.sessions.get("session-b").bank

    a.transfer(10_000, "sarah", "checking")
    assert a.balances["checking"] == 238_350
    assert b.balances["checking"] == 248_350, "one session spent another's money"


def test_ending_a_session_forgets_it(client):
    sid = "ending-session"
    client.post("/v1/chat", json={"text": "hello", "session_id": sid})
    services = client.app.state.services
    services.sessions.get(sid).bank.transfer(1_000, "sarah", "checking")

    assert client.delete(f"/v1/session/{sid}").json()["status"] == "ended"
    # a fresh conversation under the same id starts from the seeded balance
    assert services.sessions.get(sid).bank.balances["checking"] == 248_350


# --- bad input ---------------------------------------------------------------


def test_an_empty_upload_is_rejected(client):
    response = client.post("/v1/converse", files={"audio": ("a.wav", b"", "audio/wav")})
    assert response.status_code == 400
    assert "No audio" in response.json()["detail"]


def test_a_non_wav_upload_is_rejected_with_a_reason(client):
    response = client.post(
        "/v1/converse", files={"audio": ("a.wav", b"not a wav file at all", "audio/wav")}
    )
    assert response.status_code == 400
    assert "Could not read" in response.json()["detail"]


def test_oversized_audio_is_refused_before_decoding(client):
    oversized = b"RIFF" + b"\x00" * (11 * 1024 * 1024)
    response = client.post("/v1/converse", files={"audio": ("a.wav", oversized, "audio/wav")})
    assert response.status_code == 413
    assert "limit" in response.json()["detail"]


def test_a_missing_audio_field_is_a_validation_error(client):
    assert client.post("/v1/converse").status_code == 422


# --- documentation -----------------------------------------------------------


def test_the_openapi_schema_documents_every_endpoint(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/ready", "/metrics"} <= set(paths)
    assert {"/v1/transcribe", "/v1/understand", "/v1/speak"} <= set(paths)
    assert {"/v1/converse", "/v1/chat"} <= set(paths)
