"""The HTTP surface, end to end, on fake engines.

The whole service is exercised with no model, no endpoint and no network: the
fake LLM client is scripted, the bank is in memory, and the TestClient runs
the real lifespan. What these tests cannot cover is the model's judgement;
what they do cover is that no amount of model behaviour can move money without
a confirmation, and that the page is still served.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient

from voicebot.api.app import create_app
from voicebot.config import AppConfig
from voicebot.llm.base import FakeLLMClient, Reply, ToolCall
from voicebot.pipeline.audio_io import encode_wav
from voicebot.schemas import AudioBuffer

TRANSFER = {"amount_cents": 5_000, "payee": "sarah", "from_account": "checking"}


def _wav(seconds: float = 1.0, rate: int = 16_000, silent: bool = False) -> bytes:
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False, dtype=np.float32)
    samples = np.zeros_like(t) if silent else (0.5 * np.sin(2 * np.pi * 220 * t))
    return encode_wav(AudioBuffer(samples=samples.astype(np.float32), sample_rate=rate))


@pytest.fixture
def client() -> Iterator[TestClient]:
    # The context manager is what runs the lifespan, which loads the engines.
    with TestClient(create_app()) as test_client:
        yield test_client


def scripted(app_client: TestClient, replies: list[Reply]) -> FakeLLMClient:
    """Replace the service's LLM with a script, for one test."""
    fake = FakeLLMClient(replies)
    app_client.app.state.services.llm = fake
    return fake


def browser_tts(app_client: TestClient) -> None:
    from voicebot.tts.browser import BrowserTTSEngine

    app_client.app.state.services.tts = BrowserTTSEngine()


# --- operational -------------------------------------------------------------


def test_health_reports_ok_and_a_version(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["version"]


def test_ready_names_the_llm_not_an_nlu(client: TestClient) -> None:
    body = client.get("/ready").json()
    assert body["ready"] is True
    assert body["asr"] == "FakeASREngine"
    assert body["llm"] == "FakeLLMClient"
    assert "nlu" not in body


def test_metrics_exposes_tool_calls_and_the_guard_hit_rate(client: TestClient) -> None:
    scripted(
        client,
        [
            Reply(tool_calls=(ToolCall("c1", "get_balance", {"account": "savings"}),)),
            Reply(text="Your savings balance is $19,020.00."),
        ],
    )
    client.post("/v1/chat", json={"text": "balance?"})

    body = client.get("/metrics").text
    assert "voicebot_turns_total 1" in body
    assert 'voicebot_tool_calls_total{tool="get_balance"} 1' in body
    assert "voicebot_guard_hits_total 0" in body


def test_the_guard_hit_counter_rises_when_a_figure_is_invented(client: TestClient) -> None:
    scripted(client, [Reply(text="Your balance is $31,415.00.")])
    client.post("/v1/chat", json={"text": "balance?"})
    assert "voicebot_guard_hits_total 1" in client.get("/metrics").text


# --- a turn ------------------------------------------------------------------


def test_a_chat_turn_returns_the_new_shape(client: TestClient) -> None:
    scripted(client, [Reply(text="Hello there.")])
    body = client.post("/v1/chat", json={"text": "hello"}).json()

    assert body["reply"] == "Hello there."
    assert body["tool_calls"] == []
    assert body["guard_hit"] is False
    assert body["session_id"]
    # A typed turn skips the microphone, so only the stages that ran report --
    # and none of v1's stages exist any more.
    stages = set(body["latency_ms"])
    assert {"agent", "tts"} <= stages
    assert not stages & {"nlu", "dialogue", "normalize"}


def test_a_tool_call_is_reported_for_the_inspector(client: TestClient) -> None:
    scripted(
        client,
        [
            Reply(tool_calls=(ToolCall("c1", "get_balance", {"account": "savings"}),)),
            Reply(text="Your savings balance is $19,020.00."),
        ],
    )
    body = client.post("/v1/chat", json={"text": "balance?"}).json()

    call = body["tool_calls"][0]
    assert call["name"] == "get_balance"
    assert call["arguments"] == {"account": "savings"}
    assert call["result"]["balance"] == "$19,020.00"


def test_converse_accepts_audio(client: TestClient) -> None:
    scripted(client, [Reply(text="Hello there.")])
    response = client.post("/v1/converse", files={"audio": ("turn.wav", _wav(), "audio/wav")})
    assert response.status_code == 200
    assert response.json()["reply"] == "Hello there."


def test_silence_is_answered_without_calling_the_model(client: TestClient) -> None:
    fake = scripted(client, [])  # any call raises
    body = client.post(
        "/v1/converse", files={"audio": ("turn.wav", _wav(silent=True), "audio/wav")}
    ).json()

    assert body["reply"]
    assert fake.calls == []


def test_a_session_id_is_issued_when_none_is_given(client: TestClient) -> None:
    scripted(client, [Reply(text="hi")])
    assert len(client.post("/v1/chat", json={"text": "hello"}).json()["session_id"]) == 32


def test_the_conversation_persists_within_a_session(client: TestClient) -> None:
    fake = scripted(client, [Reply(text="one"), Reply(text="two")])
    first = client.post("/v1/chat", json={"text": "hello"}).json()
    client.post("/v1/chat", json={"text": "again", "session_id": first["session_id"]})

    assert len(fake.calls[1][0]) > len(fake.calls[0][0])


# --- speech out --------------------------------------------------------------


def test_browser_tts_returns_no_audio_but_asks_the_page_to_speak(client: TestClient) -> None:
    browser_tts(client)
    scripted(client, [Reply(text="Hello there.")])
    body = client.post("/v1/chat", json={"text": "hello"}).json()

    assert body["audio_b64"] is None
    assert body["speak_client_side"] is True


def test_a_server_side_engine_still_returns_audio(client: TestClient) -> None:
    scripted(client, [Reply(text="Hello there.")])
    body = client.post("/v1/chat", json={"text": "hello"}).json()

    assert body["audio_b64"]
    assert body["speak_client_side"] is False


# --- confirmation ------------------------------------------------------------


def test_confirm_executes_the_pending_action(client: TestClient) -> None:
    scripted(
        client,
        [
            Reply(tool_calls=(ToolCall("c1", "transfer_money", TRANSFER),)),
            Reply(text="Transfer $50.00 to sarah from checking. Shall I go ahead?"),
            Reply(text="Done. $50.00 sent to sarah."),
        ],
    )
    bank = client.app.state.services.bank
    session_id = client.post("/v1/chat", json={"text": "send sarah fifty"}).json()["session_id"]
    assert bank.balance("checking") == 248_350  # nothing yet

    body = client.post("/v1/confirm", json={"session_id": session_id}).json()

    assert body["reply"]
    assert bank.balance("checking") == 243_350


def test_cancel_leaves_the_money_alone(client: TestClient) -> None:
    scripted(
        client,
        [
            Reply(tool_calls=(ToolCall("c1", "transfer_money", TRANSFER),)),
            Reply(text="Transfer $50.00 to sarah. Shall I go ahead?"),
            Reply(text="No problem, cancelled."),
        ],
    )
    bank = client.app.state.services.bank
    session_id = client.post("/v1/chat", json={"text": "send sarah fifty"}).json()["session_id"]

    client.post("/v1/cancel", json={"session_id": session_id})

    assert bank.balance("checking") == 248_350


def test_confirm_with_no_pending_action_answers_cleanly(client: TestClient) -> None:
    # The user clicks Yes twice, or clicks it after a restart.
    scripted(client, [Reply(text="There wasn't anything waiting.")])
    response = client.post("/v1/confirm", json={"session_id": "fresh"})

    assert response.status_code == 200
    assert response.json()["reply"]


def test_one_session_cannot_confirm_anothers_transfer(client: TestClient) -> None:
    scripted(
        client,
        [
            Reply(tool_calls=(ToolCall("c1", "transfer_money", TRANSFER),)),
            Reply(text="Shall I go ahead?"),
            Reply(text="Nothing was pending."),
        ],
    )
    bank = client.app.state.services.bank
    client.post("/v1/chat", json={"text": "send sarah fifty", "session_id": "tab-one"})

    client.post("/v1/confirm", json={"session_id": "tab-two"})

    assert bank.balance("checking") == 248_350


# --- housekeeping ------------------------------------------------------------


def test_reset_restores_the_seeded_balances(client: TestClient) -> None:
    bank = client.app.state.services.bank
    bank.transfer(5_000, "sarah", "checking")

    assert client.post("/v1/reset", json={}).json()["status"] == "reset"
    assert bank.balance("checking") == 248_350


def test_reset_keeps_the_audit_log(client: TestClient) -> None:
    scripted(client, [Reply(text="hi")])
    client.post("/v1/chat", json={"text": "hello"})
    before = client.app.state.services.bank.turn_count()

    client.post("/v1/reset", json={})

    assert client.app.state.services.bank.turn_count() == before


def test_ending_a_session_forgets_it(client: TestClient) -> None:
    scripted(client, [Reply(text="hi")])
    session_id = client.post("/v1/chat", json={"text": "hello"}).json()["session_id"]
    assert client.delete(f"/v1/session/{session_id}").json()["status"] == "ended"


def test_the_per_stage_nlu_route_is_gone(client: TestClient) -> None:
    # The static mount at the root answers an unrouted POST with 405 rather
    # than 404; either way nothing handles it any more.
    assert client.post("/v1/understand", json={"text": "x"}).status_code != 200


# --- the front end -----------------------------------------------------------


def test_the_page_is_served_at_the_root(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "<!doctype html>" in response.text.lower()


@pytest.mark.parametrize("asset", ["app.css", "app.js", "vad.js"])
def test_every_asset_the_page_asks_for_is_served(client: TestClient, asset: str) -> None:
    assert client.get(f"/{asset}").status_code == 200


def test_the_api_still_wins_over_the_static_mount(client: TestClient) -> None:
    # The static files are mounted at the root, so an API path that fell
    # through to them would 404 instead of answering.
    assert client.get("/health").json()["status"] == "ok"


def test_a_bare_config_starts_with_fakes_and_an_in_memory_bank() -> None:
    config = AppConfig()
    assert config.llm.engine == "fake"
    assert config.database.path == ":memory:"


def test_the_payload_says_when_a_confirmation_is_waiting(client: TestClient) -> None:
    # The page shows its Yes/Cancel buttons from this flag. v1 read it off the
    # dialogue action type, which no longer exists.
    scripted(
        client,
        [
            Reply(tool_calls=(ToolCall("c1", "transfer_money", TRANSFER),)),
            Reply(text="Transfer $50.00 to sarah. Shall I go ahead?"),
            Reply(text="Done."),
        ],
    )
    pending = client.post("/v1/chat", json={"text": "send sarah fifty"}).json()
    assert pending["awaiting_confirmation"] is True

    settled = client.post("/v1/confirm", json={"session_id": pending["session_id"]}).json()
    assert settled["awaiting_confirmation"] is False


def test_an_ordinary_turn_is_not_awaiting_confirmation(client: TestClient) -> None:
    scripted(client, [Reply(text="Hello there.")])
    assert client.post("/v1/chat", json={"text": "hello"}).json()["awaiting_confirmation"] is False
