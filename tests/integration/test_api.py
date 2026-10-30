"""The HTTP surface, end to end, on fake engines.

The whole service is exercised with no model, no endpoint and no network: the
fake LLM client is scripted, the bank is in memory, and the TestClient runs
the real lifespan. What these tests cannot cover is the model's judgement;
what they do cover is that no amount of model behaviour can move money without
a confirmation, and that the page is still served.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests import support
from voicebot.api.app import STATIC_DIR, create_app
from voicebot.api.deps import Services
from voicebot.api.sessions import InMemorySessionStore
from voicebot.bank.sqlite_bank import DEMO_PIN
from voicebot.config import AppConfig
from voicebot.llm.base import FakeLLMClient, Reply, ToolCall
from voicebot.pipeline.audio_io import encode_wav
from voicebot.schemas import AudioBuffer
from voicebot.tts.base import FakeTTSEngine

# Named by what is on disk rather than by a list kept here, so a module
# added to the page cannot be left out of the tests that serve it.
ASSETS = sorted(p.name for p in STATIC_DIR.iterdir() if p.suffix in {".js", ".css"})

TRANSFER = {"amount_cents": 5_000, "payee": "sarah", "from_account": "checking"}
MOVE = {"amount_cents": 5_000, "from_account": "checking", "to_account": "savings"}


def _wav(seconds: float = 1.0, rate: int = 16_000, silent: bool = False) -> bytes:
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False, dtype=np.float32)
    samples = np.zeros_like(t) if silent else (0.5 * np.sin(2 * np.pi * 220 * t))
    return encode_wav(AudioBuffer(samples=samples.astype(np.float32), sample_rate=rate))


@pytest.fixture
def client() -> Iterator[TestClient]:
    # The context manager is what runs the lifespan, which loads the engines.
    with TestClient(create_app()) as test_client:
        if not support.TESTING_POSTGRES:
            yield test_client
            return
        # The same swap `scripted` makes for the model: every route reaches
        # the bank through `services.bank` or a session made from it.
        with support.fresh_postgres_bank() as bank:
            services = test_client.app.state.services
            services.bank = bank
            services.sessions = InMemorySessionStore(bank=bank)
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
    """Still counted, now that the loop sends it to look the figure up first.

    The lookup is the better answer when it works. When the model invents the
    same figure anyway, the guard stops it and the counter has to notice --
    that number is how anyone sees the model reaching for figures at all.
    """
    scripted(
        client,
        [
            Reply(text="Your balance is $31,415.00."),
            Reply(tool_calls=(ToolCall("c1", "list_cards", {}),)),
            Reply(text="Your balance is $31,415.00."),
        ],
    )
    client.post("/v1/chat", json={"text": "balance?"})
    assert client.app.state.services.audit.flush()

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
    # Turn records are written after the turn, on a queue; wait for them.
    assert client.app.state.services.audit.flush()
    bank = client.app.state.services.bank
    session_id = client.post("/v1/chat", json={"text": "send sarah fifty"}).json()["session_id"]
    assert bank.balance("checking") == 248_350  # nothing yet

    body = client.post("/v1/confirm", json={"pin": DEMO_PIN, "session_id": session_id}).json()

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
    # Turn records are written after the turn, on a queue; wait for them.
    assert client.app.state.services.audit.flush()
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
    # Turn records are written after the turn, on a queue; wait for them.
    assert client.app.state.services.audit.flush()
    bank = client.app.state.services.bank
    client.post("/v1/chat", json={"text": "send sarah fifty", "session_id": "tab-one"})

    client.post("/v1/confirm", json={"session_id": "tab-two"})

    assert bank.balance("checking") == 248_350


# --- housekeeping ------------------------------------------------------------


def test_reset_restores_the_seeded_balances(client: TestClient) -> None:
    # Turn records are written after the turn, on a queue; wait for them.
    assert client.app.state.services.audit.flush()
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


@pytest.mark.parametrize("asset", ASSETS)
def test_every_asset_the_page_asks_for_is_served(client: TestClient, asset: str) -> None:
    assert client.get(f"/{asset}").status_code == 200


def test_the_page_asks_for_nothing_that_is_not_there(client: TestClient) -> None:
    """Every stylesheet and script the page links must be served.

    The stylesheets are split by screen and loaded in order, which is what
    keeps the cascade right. A link that 404s does not fail anything: the
    page still loads, one screen's rules are simply missing.
    """
    page = client.get("/").text
    asked = re.findall(r'(?:href|src)="(?!https?:|data:)([^"]+)"', page)
    assert asked, "the page links nothing, which cannot be right"
    for path in asked:
        assert client.get(f"/{path}").status_code == 200, f"the page asks for {path}"


def test_every_module_import_points_at_a_file_that_exists() -> None:
    """A missing module is a blank page, and nothing else says so.

    The browser resolves these paths itself and reports the failure only to
    its own console, so a renamed or forgotten module leaves a page that
    loads, draws nothing, and raises no error anyone sees.
    """
    for module in sorted(STATIC_DIR.glob("*.js")):
        source = module.read_text(encoding="utf-8")
        for target in re.findall(r'^import[\s\S]*?from "\./([\w.-]+)";', source, re.M):
            assert (STATIC_DIR / target).is_file(), f"{module.name} imports missing {target}"


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

    settled = client.post(
        "/v1/confirm", json={"pin": DEMO_PIN, "session_id": pending["session_id"]}
    ).json()
    assert settled["awaiting_confirmation"] is False


def test_an_ordinary_turn_is_not_awaiting_confirmation(client: TestClient) -> None:
    scripted(client, [Reply(text="Hello there.")])
    assert client.post("/v1/chat", json={"text": "hello"}).json()["awaiting_confirmation"] is False


def test_the_confirmed_action_reaches_the_inspector(client: TestClient) -> None:
    # The one action that actually moved money was rendering as "no tools
    # called", because the confirm path produced a result with no matching call.
    scripted(
        client,
        [
            Reply(tool_calls=(ToolCall("c1", "transfer_money", TRANSFER),)),
            Reply(text="Shall I go ahead?"),
            Reply(text="Done."),
        ],
    )
    sid = client.post("/v1/chat", json={"text": "send sarah fifty"}).json()["session_id"]

    body = client.post("/v1/confirm", json={"pin": DEMO_PIN, "session_id": sid}).json()

    assert body["tool_calls"], "the confirmed transfer is missing from the inspector"
    assert body["tool_calls"][0]["name"] == "transfer_money"
    assert body["tool_calls"][0]["result"]["status"] == "done"


def test_the_confirmation_token_is_not_published(client: TestClient) -> None:
    scripted(
        client,
        [
            Reply(tool_calls=(ToolCall("c1", "transfer_money", TRANSFER),)),
            Reply(text="Shall I go ahead?"),
        ],
    )
    body = client.post("/v1/chat", json={"text": "send sarah fifty"}).json()
    for call in body["tool_calls"]:
        assert "confirmation_token" not in call["arguments"]
        assert "token" not in call["result"]


def test_a_turn_with_no_tools_logs_null_rather_than_an_empty_array(
    client: TestClient,
) -> None:
    scripted(client, [Reply(text="Hello there.")])
    client.post("/v1/chat", json={"text": "hello"})

    # Turn records are written after the turn, on a queue; wait for them.
    assert client.app.state.services.audit.flush()
    bank = client.app.state.services.bank
    row = support.rows(bank, "SELECT tool_calls FROM turns ORDER BY id DESC LIMIT 1")[0]
    assert row["tool_calls"] is None


def test_an_unknown_engine_name_is_refused_rather_than_silently_faked() -> None:
    # A single typo in a deploy would otherwise start cleanly and answer every
    # question with "the model isn't configured yet".
    from voicebot.api.deps import build_services
    from voicebot.config import AppConfig, ASRConfig, DatabaseConfig, LLMConfig, TTSConfig

    for config in (
        AppConfig(asr=ASRConfig(engine="whispr")),
        AppConfig(llm=LLMConfig(engine="openapi")),
        AppConfig(tts=TTSConfig(engine="vit")),
        AppConfig(database=DatabaseConfig(engine="postgress")),
    ):
        with pytest.raises(ValueError, match="unknown"):
            build_services(config=config)


def test_the_office_endpoints_build_as_the_engines() -> None:
    # Building opens no connection, so this needs no network: the office
    # recognisers and voices are first called on the first turn.
    from voicebot.api.deps import build_services
    from voicebot.asr.endpoints import EndpointsASREngine
    from voicebot.config import AppConfig, ASRConfig, TTSConfig
    from voicebot.tts.endpoints import EndpointsTTSEngine

    services = build_services(
        config=AppConfig(asr=ASRConfig(engine="endpoints"), tts=TTSConfig(engine="endpoints"))
    )

    assert isinstance(services.asr, EndpointsASREngine)
    assert isinstance(services.tts, EndpointsTTSEngine)
    # The page plays what the office voice made, rather than speaking itself.
    assert services.speaks_client_side is False


def test_the_serving_config_uses_the_office_endpoints() -> None:
    from voicebot.config import load_config

    config = load_config("configs/serve.yaml")
    assert (config.asr.engine, config.tts.engine) == ("endpoints", "endpoints")


def test_the_committed_default_config_still_loads() -> None:
    # configs/app.yaml is what the docs point at; it must not rot.
    from voicebot.config import load_config

    config = load_config("configs/app.yaml")
    assert config.llm.engine
    assert config.database.path


@pytest.mark.parametrize("asset", ["/", *(f"/{name}" for name in ASSETS)])
def test_the_page_and_its_assets_must_be_revalidated(client: TestClient, asset: str) -> None:
    """Without this the browser runs yesterday's JavaScript.

    FastAPI's StaticFiles sends an ETag but no Cache-Control, so a browser
    applies heuristic caching and may not ask whether the file changed. After
    an update that looks exactly like the fix not working -- which is how this
    was found.
    """
    response = client.get(asset)
    assert response.status_code == 200
    assert "no-cache" in response.headers.get("cache-control", "")


# --- the account state travels with every turn -------------------------------
#
# Both the balances on screen and the figures the bot speaks have to come from
# the same answer, or the page can show one number while the bot says another.


def test_a_turn_carries_the_current_balances(client: TestClient) -> None:
    scripted(client, [Reply(text="Hello there.")])
    body = client.post("/v1/chat", json={"text": "hello"}).json()

    accounts = {a["name"]: a for a in body["accounts"]}
    assert accounts["checking"]["balance_cents"] == 248_350
    assert accounts["checking"]["balance"] == "$2,483.50"
    assert accounts["savings"]["balance_cents"] == 1_902_000


def test_a_turn_carries_the_bills(client: TestClient) -> None:
    scripted(client, [Reply(text="Hello there.")])
    bills = {b["name"]: b for b in client.post("/v1/chat", json={"text": "hi"}).json()["bills"]}

    assert bills["electricity"]["amount"] == "$72.50"
    assert bills["electricity"]["paid"] is False


def test_the_balances_reflect_the_turn_that_just_happened(client: TestClient) -> None:
    """The page must never show a figure the turn has already changed."""
    scripted(
        client,
        [
            Reply(tool_calls=(ToolCall("c1", "transfer_money", TRANSFER),)),
            Reply(text="Shall I go ahead?"),
            Reply(text="Done."),
        ],
    )
    sid = client.post("/v1/chat", json={"text": "send sarah fifty"}).json()["session_id"]

    settled = client.post("/v1/confirm", json={"pin": DEMO_PIN, "session_id": sid}).json()

    checking = next(a for a in settled["accounts"] if a["name"] == "checking")
    assert checking["balance_cents"] == 243_350
    assert checking["balance"] == "$2,433.50"


def test_paying_a_bill_shows_in_the_same_answer(client: TestClient) -> None:
    scripted(
        client,
        [
            Reply(tool_calls=(ToolCall("c1", "pay_bill", {"bill": "water"}),)),
            Reply(text="Shall I go ahead?"),
            Reply(text="Paid."),
        ],
    )
    sid = client.post("/v1/chat", json={"text": "pay the water bill"}).json()["session_id"]

    settled = client.post("/v1/confirm", json={"pin": DEMO_PIN, "session_id": sid}).json()

    water = next(b for b in settled["bills"] if b["name"] == "water")
    assert water["paid"] is True


# --- the opening ledger ------------------------------------------------------
#
# A turn carries the balances with it, but the page has to show them before
# there has been a turn. That first paint is a read, not a conversation: it
# must not reach the model, spend a session, or count against the turn total.


def test_the_ledger_can_be_read_without_a_turn(client: TestClient) -> None:
    """No script is installed, so any call to the model raises instead."""
    body = client.get("/v1/accounts").json()

    accounts = {a["name"]: a for a in body["accounts"]}
    assert accounts["checking"]["balance_cents"] == 248_350
    assert accounts["checking"]["balance"] == "$2,483.50"

    bills = {b["name"]: b for b in body["bills"]}
    assert bills["electricity"]["amount"] == "$72.50"
    assert bills["electricity"]["paid"] is False


def test_reading_the_ledger_is_not_a_turn(client: TestClient) -> None:
    before = client.get("/metrics").text

    client.get("/v1/accounts")

    assert client.get("/metrics").text == before


def test_the_ledger_shows_what_the_last_turn_did(client: TestClient) -> None:
    scripted(
        client,
        [
            Reply(tool_calls=(ToolCall("c1", "transfer_money", TRANSFER),)),
            Reply(text="Shall I go ahead?"),
            Reply(text="Done."),
        ],
    )
    sid = client.post("/v1/chat", json={"text": "send sarah fifty"}).json()["session_id"]
    client.post("/v1/confirm", json={"pin": DEMO_PIN, "session_id": sid})

    ledger = client.get("/v1/accounts").json()
    checking = next(a for a in ledger["accounts"] if a["name"] == "checking")
    assert checking["balance_cents"] == 243_350


# --- handing the conversation to a person ------------------------------------


def test_a_turn_reports_the_case_it_opened(client: TestClient) -> None:
    scripted(client, [Reply(text="I can sort that out for you myself.")])

    body = client.post("/v1/chat", json={"text": "I want to speak to a human"}).json()

    assert body["handed_off"]
    assert body["handed_off"] in body["reply"]


def test_an_ordinary_turn_reports_no_case(client: TestClient) -> None:
    scripted(client, [Reply(text="Hello there.")])
    body = client.post("/v1/chat", json={"text": "hello"}).json()

    assert body["handed_off"] is None


def test_the_case_knows_which_conversation_raised_it(client: TestClient) -> None:
    """Without the real session id a specialist cannot find the transcript."""
    scripted(client, [Reply(text="I can sort that out myself.")])

    body = client.post("/v1/chat", json={"text": "put me through to a person"}).json()

    # Turn records are written after the turn, on a queue; wait for them.
    assert client.app.state.services.audit.flush()
    bank = client.app.state.services.bank
    case = bank.handoffs(session_id=body["session_id"])
    assert case and case[0].reference == body["handed_off"]
    assert case[0].turn_index == 1


def test_a_turn_reports_a_dispute_it_filed(client: TestClient) -> None:
    claim = {"transaction_id": 1, "reason": "not recognised"}
    scripted(
        client,
        [
            Reply(tool_calls=(ToolCall("c1", "dispute_transaction", claim),)),
            Reply(text="Shall I raise that?"),
            Reply(text="Filed."),
        ],
    )
    sid = client.post("/v1/chat", json={"text": "I don't know that charge"}).json()["session_id"]

    body = client.post("/v1/confirm", json={"session_id": sid}).json()

    assert body["dispute_filed"]
    assert body["handed_off"] is None, "a dispute is filed, not a handoff"


def test_an_ordinary_turn_files_no_dispute(client: TestClient) -> None:
    scripted(client, [Reply(text="Hello there.")])
    assert client.post("/v1/chat", json={"text": "hi"}).json()["dispute_filed"] is None


# --- the transactions screen -------------------------------------------------


def test_the_statement_endpoint_returns_a_month(client: TestClient) -> None:
    body = client.get("/v1/statement").json()

    assert body["rows"]
    assert body["month_label"]
    assert body["total_balance"] == "$21,503.50"
    assert body["outgoing_cents"] > 0


def test_a_statement_row_carries_what_the_screen_shows(client: TestClient) -> None:
    row = client.get("/v1/statement").json()["rows"][0]

    shown = {"id", "merchant", "account", "amount", "amount_cents", "direction", "occurred_at"}
    assert shown <= set(row)
    assert row["amount_cents"] > 0, "amounts are positive; direction says which way"


def test_a_statement_can_be_asked_for_by_month(client: TestClient) -> None:
    body = client.get("/v1/statement", params={"year": 1999, "month": 1}).json()

    assert body["rows"] == []
    assert body["incoming_cents"] == 0


def test_an_impossible_month_is_a_bad_request(client: TestClient) -> None:
    assert client.get("/v1/statement", params={"year": 2026, "month": 13}).status_code == 400


def test_reading_a_statement_is_not_a_turn(client: TestClient) -> None:
    before = client.get("/metrics").text
    client.get("/v1/statement")

    assert client.get("/metrics").text == before


def test_a_statement_row_says_whether_it_was_a_transfer(client: TestClient) -> None:
    """ "Checking / Savings" reads as neither one thing nor the other.

    A row whose counterparty is one of the customer's own accounts is a
    transfer between them, and the screen has to word it differently from a
    shop. The server decides, because only the server knows which names are
    accounts.
    """
    scripted(
        client,
        [
            Reply(tool_calls=(ToolCall("c1", "move_money", MOVE),)),
            Reply(text="Shall I?"),
            Reply(text="Done."),
        ],
    )
    sid = client.post("/v1/chat", json={"text": "move fifty to savings"}).json()["session_id"]
    client.post("/v1/confirm", json={"pin": DEMO_PIN, "session_id": sid})

    rows = client.get("/v1/statement").json()["rows"]
    assert [row["kind"] for row in rows[:2]] == ["transfer", "transfer"]
    assert all(row["kind"] == "merchant" for row in rows[2:])


# --- the cards screen --------------------------------------------------------


def test_the_cards_endpoint_draws_both_cards(client: TestClient) -> None:
    cards = client.get("/v1/cards").json()["cards"]

    assert {card["last4"] for card in cards} == {"4417", "9021"}
    assert {card["kind"] for card in cards} == {"debit", "credit"}


def test_a_card_never_leaves_with_more_than_four_digits(client: TestClient) -> None:
    """The bank holds four digits, so four is all that can be published."""
    for card in client.get("/v1/cards").json()["cards"]:
        assert len(card["last4"]) == 4
        assert card["masked"].endswith(card["last4"])
        assert card["masked"].count("•") >= 12


def test_a_card_carries_its_controls(client: TestClient) -> None:
    card = client.get("/v1/cards").json()["cards"][0]

    assert [control["name"] for control in card["controls"]] == ["ecommerce", "retail", "atm"]
    assert all(control["label"] for control in card["controls"])
    assert all(isinstance(control["enabled"], bool) for control in card["controls"])


def test_reading_the_cards_is_not_a_turn(client: TestClient) -> None:
    before = client.get("/metrics").text
    client.get("/v1/cards")

    assert client.get("/metrics").text == before


# --- the monthly screen ------------------------------------------------------


def test_the_monthly_endpoint_groups_by_category(client: TestClient) -> None:
    body = client.get("/v1/monthly").json()

    assert [group["category"] for group in body["groups"]] == [
        "rent",
        "bills",
        "wifi",
        "groceries",
        "subscriptions",
    ]
    assert all(group["label"] for group in body["groups"])


def test_the_monthly_totals_add_up(client: TestClient) -> None:
    body = client.get("/v1/monthly").json()

    from_groups = sum(group["total_cents"] for group in body["groups"])
    assert body["committed_cents"] == from_groups
    assert body["paid_cents"] + body["remaining_cents"] == body["committed_cents"]


def test_nothing_is_paid_to_begin_with(client: TestClient) -> None:
    body = client.get("/v1/monthly").json()

    assert body["paid_cents"] == 0
    assert body["remaining_cents"] == body["committed_cents"]


def test_paying_one_moves_it_from_remaining_to_paid(client: TestClient) -> None:
    before = client.get("/v1/monthly").json()
    scripted(
        client,
        [
            Reply(tool_calls=(ToolCall("c1", "pay_bill", {"bill": "electricity"}),)),
            Reply(text="Shall I?"),
            Reply(text="Paid."),
        ],
    )
    sid = client.post("/v1/chat", json={"text": "pay the electricity"}).json()["session_id"]
    client.post("/v1/confirm", json={"pin": DEMO_PIN, "session_id": sid})

    after = client.get("/v1/monthly").json()
    assert after["paid_cents"] == 7_250
    assert after["remaining_cents"] == before["remaining_cents"] - 7_250
    assert after["committed_cents"] == before["committed_cents"]


def test_a_plan_says_when_it_falls_due(client: TestClient) -> None:
    plan = client.get("/v1/monthly").json()["groups"][0]["plans"][0]

    assert {"name", "amount", "amount_cents", "due_day", "due_label", "paid"} <= set(plan)
    assert plan["due_label"]


# --- acting from a screen ----------------------------------------------------
#
# The card buttons are not a second way to change an account. They arm the
# same pending action the spoken path arms, and the same /v1/confirm settles
# it -- so a click and a sentence go through one gate, and the model is told
# either way.


def test_arming_from_a_screen_changes_nothing_yet(client: TestClient) -> None:
    body = client.post(
        "/v1/action",
        json={"tool": "freeze_card", "arguments": {"last4": "4417"}, "reason": "left it in a taxi"},
    ).json()

    assert body["armed"] is True
    assert "4417" in body["summary"]
    assert not any(c["frozen"] for c in client.get("/v1/cards").json()["cards"])


def test_confirming_a_screen_action_settles_it(client: TestClient) -> None:
    scripted(client, [Reply(text="That card is frozen now.")])
    armed = client.post(
        "/v1/action",
        json={"tool": "freeze_card", "arguments": {"last4": "4417"}, "reason": "lost"},
    ).json()

    client.post("/v1/confirm", json={"session_id": armed["session_id"]})

    frozen = {c["last4"] for c in client.get("/v1/cards").json()["cards"] if c["frozen"]}
    assert frozen == {"4417"}


def test_replacing_from_a_screen_sends_another_card(client: TestClient) -> None:
    scripted(client, [Reply(text="Done.")])
    before = len(client.get("/v1/cards").json()["cards"])
    armed = client.post(
        "/v1/action",
        json={
            "tool": "freeze_card",
            "arguments": {"last4": "4417", "order_replacement": True},
            "reason": "the chip stopped working",
        },
    ).json()
    assert "replacement" in armed["summary"]

    client.post("/v1/confirm", json={"session_id": armed["session_id"]})

    assert len(client.get("/v1/cards").json()["cards"]) == before + 1


def test_a_screen_can_only_run_what_a_screen_offers(client: TestClient) -> None:
    """The endpoint is not a way to run any tool with any arguments."""
    response = client.post(
        "/v1/action",
        json={
            "tool": "transfer_money",
            "arguments": {"amount_cents": 5_000, "payee": "sarah", "from_account": "checking"},
            "reason": "trying it on",
        },
    )

    assert response.status_code == 400
    assert client.app.state.services.bank.balance("checking") == 248_350


def test_a_screen_action_has_to_say_why(client: TestClient) -> None:
    response = client.post(
        "/v1/action",
        json={"tool": "freeze_card", "arguments": {"last4": "4417"}, "reason": "   "},
    )

    assert response.status_code == 422


def test_a_refusal_comes_back_as_a_sentence_not_a_crash(client: TestClient) -> None:
    body = client.post(
        "/v1/action",
        json={"tool": "freeze_card", "arguments": {"last4": "0000"}, "reason": "lost"},
    ).json()

    assert body["armed"] is False
    assert body["error"]


def test_the_reason_reaches_the_audit_log(client: TestClient) -> None:
    """Typing why is not theatre: it is written down with the action."""
    client.post(
        "/v1/action",
        json={"tool": "freeze_card", "arguments": {"last4": "4417"}, "reason": "left it in a taxi"},
    )

    # Turn records are written after the turn, on a queue; wait for them.
    assert client.app.state.services.audit.flush()
    bank = client.app.state.services.bank
    rows = support.rows(bank, "SELECT user_text, bot_text FROM turns")
    assert any("taxi" in (row["user_text"] or "") for row in rows)


def test_a_screen_action_puts_what_was_asked_into_the_conversation(client: TestClient) -> None:
    """Found live, not here: confirming raised 400, "no user query found".

    Arming from a screen left the conversation empty, so settling it sent the
    model a history holding a tool call and its result and nothing a customer
    had ever said. The typed reason is what belongs there -- it is the only
    record of why, and the model needs it to say what happened.
    """
    client.post(
        "/v1/action",
        json={
            "tool": "freeze_card",
            "arguments": {"last4": "4417"},
            "reason": "left it on the bus",
        },
    )

    session = next(iter(client.app.state.services.sessions._entries.values())).session
    spoken = [m for m in session.conversation.messages if m.get("role") == "user"]
    assert spoken, "the conversation has to hold what the customer asked for"
    assert "bus" in spoken[-1]["content"]
    assert "4417" in spoken[-1]["content"]


# --- the details screen, behind a PIN ----------------------------------------
#
# Nothing printed on a card is readable without it, and the PIN itself never
# travels except to be checked. A wrong one is counted; three lock the card.


def details(client: TestClient, pin: str, last4: str = "4417"):
    return client.post(f"/v1/cards/{last4}/details", json={"pin": pin})


def test_the_right_pin_shows_what_is_on_the_card(client: TestClient) -> None:
    body = details(client, "1234").json()

    assert body["holder"] == "Rehan Ali Ch"
    groups = body["number"].split()
    assert len(groups) == 4 and groups[-1] == "4417"
    assert len(body["security_code"]) == 3
    assert body["status"] == "Active"


def test_a_wrong_pin_shows_nothing(client: TestClient) -> None:
    response = details(client, "9999")

    assert response.status_code == 403
    assert "number" not in response.json()


def test_three_wrong_pins_lock_the_card(client: TestClient) -> None:
    for _ in range(3):
        details(client, "9999")

    locked = details(client, "1234")
    assert locked.status_code == 423
    assert "locked" in locked.json()["detail"].lower()


def test_the_security_code_is_different_every_time(client: TestClient) -> None:
    seen = {details(client, "1234").json()["security_code"] for _ in range(12)}
    assert len(seen) > 1


def test_a_card_that_is_not_there_has_no_details(client: TestClient) -> None:
    assert details(client, "1234", last4="0000").status_code == 404


def test_the_card_list_still_shows_only_four_digits(client: TestClient) -> None:
    """The list needs no PIN, because it carries nothing that needs one."""
    for card in client.get("/v1/cards").json()["cards"]:
        assert "number" not in card
        assert card["masked"].count("•") >= 12


def test_reading_details_is_not_a_turn(client: TestClient) -> None:
    before = client.get("/metrics").text
    details(client, "1234")

    assert client.get("/metrics").text == before


# --- changing the PIN --------------------------------------------------------


def test_the_pin_can_be_changed_with_the_old_one(client: TestClient) -> None:
    response = client.post("/v1/cards/4417/pin", json={"current_pin": "1234", "new_pin": "8765"})

    assert response.status_code == 200
    assert details(client, "8765").status_code == 200
    assert details(client, "1234").status_code == 403


def test_the_pin_cannot_be_changed_without_the_old_one(client: TestClient) -> None:
    response = client.post("/v1/cards/4417/pin", json={"current_pin": "0000", "new_pin": "8765"})

    assert response.status_code == 403
    assert details(client, "1234").status_code == 200


def test_a_new_pin_has_to_be_four_digits(client: TestClient) -> None:
    for bad in ("123", "12345", "abcd"):
        response = client.post("/v1/cards/4417/pin", json={"current_pin": "1234", "new_pin": bad})
        assert response.status_code == 422, bad


def test_a_pin_never_reaches_the_audit_log(client: TestClient) -> None:
    """The one thing that must not be written down anywhere."""
    details(client, "1234")
    client.post("/v1/cards/4417/pin", json={"current_pin": "1234", "new_pin": "8765"})

    # Turn records are written after the turn, on a queue; wait for them.
    assert client.app.state.services.audit.flush()
    bank = client.app.state.services.bank
    written = " ".join(
        str(value) for row in support.rows(bank, "SELECT * FROM turns") for value in row.values()
    )
    assert "1234" not in written and "8765" not in written


def test_a_pin_can_be_checked_without_showing_anything(client: TestClient) -> None:
    """What the reset flow asks before it offers a new PIN.

    Deliberately not the details endpoint: checking whether somebody knows
    the PIN should not hand back the card number as a side effect.
    """
    good = client.post("/v1/cards/4417/pin/check", json={"pin": "1234"})
    assert good.status_code == 200
    assert "number" not in good.json()

    assert client.post("/v1/cards/4417/pin/check", json={"pin": "9999"}).status_code == 403


# --- the PIN in front of money -----------------------------------------------
#
# The page walks this in two presses: Yes, which comes back asking for the
# PIN, and then the PIN, which approves and performs in one call.


def _armed_transfer(client: TestClient) -> str:
    scripted(
        client,
        [
            Reply(tool_calls=(ToolCall("c1", "transfer_money", TRANSFER),)),
            Reply(text="Transfer $50.00 to sarah from checking. Shall I go ahead?"),
            Reply(text="Done. $50.00 sent to sarah."),
        ],
    )
    return str(client.post("/v1/chat", json={"text": "send sarah fifty"}).json()["session_id"])


def test_yes_comes_back_asking_for_the_pin(client: TestClient) -> None:
    # Turn records are written after the turn, on a queue; wait for them.
    assert client.app.state.services.audit.flush()
    bank = client.app.state.services.bank
    sid = _armed_transfer(client)

    body = client.post("/v1/confirm", json={"session_id": sid}).json()

    assert body["pin_request"]["last4"] == "4417"
    assert body["pin_request"]["purpose"] == "payment"
    # Still waiting, so the bar stays and the keypad can be cancelled.
    assert body["awaiting_confirmation"] is True
    assert bank.balance("checking") == 248_350


def test_the_keypad_is_told_what_it_is_approving(client: TestClient) -> None:
    """The dialogue covers the bar, so it carries the summary itself."""
    sid = _armed_transfer(client)

    body = client.post("/v1/confirm", json={"session_id": sid}).json()

    assert "$50.00" in body["pin_request"]["summary"]
    assert "sarah" in body["pin_request"]["summary"].lower()


def test_asking_for_the_pin_does_not_reach_the_model(client: TestClient) -> None:
    """Nothing to add to "enter your PIN", and the wait would be the model's."""
    scripted(
        client,
        [
            Reply(tool_calls=(ToolCall("c1", "pay_bill", {"bill": "internet"}),)),
            Reply(text="Pay your internet bill, $40.00. Shall I go ahead?"),
            # Nothing further: a third call would exhaust the script and raise.
        ],
    )
    sid = client.post("/v1/chat", json={"text": "pay my internet bill"}).json()["session_id"]

    body = client.post("/v1/confirm", json={"session_id": sid}).json()

    assert body["pin_request"]["purpose"] == "payment"
    assert body["reply"]


def test_the_right_pin_lets_the_payment_through(client: TestClient) -> None:
    # Turn records are written after the turn, on a queue; wait for them.
    assert client.app.state.services.audit.flush()
    bank = client.app.state.services.bank
    sid = _armed_transfer(client)
    client.post("/v1/confirm", json={"session_id": sid})

    body = client.post("/v1/confirm", json={"session_id": sid, "pin": DEMO_PIN}).json()

    assert body["reply"]
    assert body["awaiting_confirmation"] is False
    assert bank.balance("checking") == 243_350


def test_a_wrong_pin_moves_nothing_and_keeps_the_agreement(client: TestClient) -> None:
    # Turn records are written after the turn, on a queue; wait for them.
    assert client.app.state.services.audit.flush()
    bank = client.app.state.services.bank
    sid = _armed_transfer(client)
    client.post("/v1/confirm", json={"session_id": sid})

    refused = client.post("/v1/confirm", json={"session_id": sid, "pin": "9999"})

    assert refused.status_code == 403
    assert bank.balance("checking") == 248_350
    # The agreement survives a wrong PIN: they try again, they do not re-agree.
    assert client.app.state.services.sessions.get(sid).executor.pending is not None


def test_the_right_pin_still_works_after_a_wrong_one(client: TestClient) -> None:
    # Turn records are written after the turn, on a queue; wait for them.
    assert client.app.state.services.audit.flush()
    bank = client.app.state.services.bank
    sid = _armed_transfer(client)
    client.post("/v1/confirm", json={"session_id": sid})
    client.post("/v1/confirm", json={"session_id": sid, "pin": "9999"})

    client.post("/v1/confirm", json={"session_id": sid, "pin": DEMO_PIN})

    assert bank.balance("checking") == 243_350


def test_a_pin_with_nothing_waiting_is_refused(client: TestClient) -> None:
    """A PIN approves a named action or it approves nothing."""
    scripted(client, [Reply(text="Nothing is pending.")])
    sid = client.post("/v1/chat", json={"text": "hello"}).json()["session_id"]

    refused = client.post("/v1/confirm", json={"session_id": sid, "pin": DEMO_PIN})

    assert refused.status_code == 409


def test_freezing_a_card_still_needs_no_pin(client: TestClient) -> None:
    """Nothing moves, so the yes is the whole of it."""
    scripted(
        client,
        [
            Reply(tool_calls=(ToolCall("c1", "freeze_card", {"last4": "4417"}),)),
            Reply(text="Freeze the card ending 4417. Shall I go ahead?"),
            Reply(text="Done, the card is frozen."),
        ],
    )
    # Turn records are written after the turn, on a queue; wait for them.
    assert client.app.state.services.audit.flush()
    bank = client.app.state.services.bank
    sid = client.post("/v1/chat", json={"text": "freeze my card"}).json()["session_id"]

    body = client.post("/v1/confirm", json={"session_id": sid}).json()

    assert body["pin_request"] is None
    assert next(c for c in bank.cards() if c.last4 == "4417").frozen is True


# --- a reply spoken a piece at a time, in the bot's own voice ------------------------


class _PiecesVoice(FakeTTSEngine):
    """Cuts at each full stop and can keep one recording, like the office voices."""

    def __init__(self, fail: bool = False) -> None:
        super().__init__()
        self.fail = fail
        self.sorry = AudioBuffer(samples=np.ones(800, dtype=np.float32), sample_rate=16_000)

    def pieces(self, text: str, voice: str | None = None) -> list[str]:
        return [p.strip() + "." for p in text.split(".") if p.strip()]

    def synthesize(self, text: str, voice: str | None = None) -> AudioBuffer:
        if self.fail:
            return AudioBuffer(samples=np.zeros(0, dtype=np.float32), sample_rate=16_000)
        return super().synthesize(text, voice)

    def kept(self, key: str, language: str) -> AudioBuffer | None:
        return self.sorry if key == "voice_unavailable" else None


def _speaking(client: TestClient, tts: _PiecesVoice, reply: str) -> None:
    services: Services = client.app.state.services  # type: ignore[attr-defined]
    services.tts = tts
    services.llm = FakeLLMClient([Reply(text=reply)])


def test_a_long_reply_arrives_a_piece_at_a_time(client: TestClient) -> None:
    _speaking(client, _PiecesVoice(), "One. Two. Three.")
    turn = client.post("/v1/chat", json={"text": "hello"}).json()

    assert turn["audio_b64"] and turn["audio_more"] == 2 and turn["audio_id"]
    base = f"/v1/sessions/{turn['session_id']}/audio/{turn['audio_id']}"
    for n in (1, 2):
        piece = client.get(f"{base}/{n}")
        assert piece.status_code == 200 and piece.headers["content-type"] == "audio/wav"
        assert piece.content[:4] == b"RIFF"
    assert client.get(f"{base}/3").status_code == 404


def test_a_new_turn_stops_what_was_left_of_the_last_one(client: TestClient) -> None:
    _speaking(client, _PiecesVoice(), "One. Two. Three.")
    first = client.post("/v1/chat", json={"text": "hello"}).json()
    services: Services = client.app.state.services  # type: ignore[attr-defined]
    services.llm = FakeLLMClient([Reply(text="Fine.")])
    client.post("/v1/chat", json={"text": "again", "session_id": first["session_id"]})

    stale = f"/v1/sessions/{first['session_id']}/audio/{first['audio_id']}/1"
    assert client.get(stale).status_code == 404


def test_a_failed_voice_is_said_in_the_bots_own_voice_never_the_browsers(
    client: TestClient,
) -> None:
    _speaking(client, _PiecesVoice(fail=True), "Your balance is fine.")
    turn = client.post("/v1/chat", json={"text": "hello"}).json()
    # The kept recording of "my voice isn't working", and no hand-over to a
    # browser voice that may be the other gender and has no Urdu.
    assert turn["audio_b64"] and turn["audio_more"] == 0
    assert turn["speak_client_side"] is False
    assert turn["reply"] == "Your balance is fine."


def test_the_page_is_told_who_is_speaking(client: TestClient) -> None:
    voice = client.get("/v1/voice").json()
    assert voice == {
        "speaks_client_side": False,
        "gender": "female",
        "name": {"en": "Layla", "ur": "لیلیٰ"},
    }


def test_the_page_can_keep_the_bots_recordings_of_its_failure_sentences(
    client: TestClient,
) -> None:
    assert client.get("/v1/messages/connection_lost/audio?language=ur").status_code == 404
    _speaking(client, _PiecesVoice(), "x")
    kept = client.get("/v1/messages/voice_unavailable/audio?language=ur")
    assert kept.status_code == 200 and kept.content[:4] == b"RIFF"


# --- callers in two languages at once; a caller who hangs up ------------------------


class _AnswersInTheirLanguage(FakeLLMClient):
    """Answers in whatever language the note beside the customer's words asks for."""

    def __init__(self) -> None:
        super().__init__([])

    def complete(self, messages, tools, tool_choice="auto"):  # type: ignore[no-untyped-def]
        time.sleep(0.05)  # long enough for the two callers' turns to overlap
        note = messages[-1]["content"]
        urdu = "Reply ONLY in Urdu" in note
        return Reply(
            text="آپ کا بیلنس ٹھیک ہے۔ اور کچھ؟" if urdu else "Your balance is fine. Anything else?"
        )


class _RecordingVoice(_PiecesVoice):
    def __init__(self) -> None:
        super().__init__()
        self.lock = threading.Lock()
        self.voices: list[tuple[str, str | None]] = []

    def synthesize(self, text: str, voice: str | None = None) -> AudioBuffer:
        with self.lock:
            self.voices.append((text, voice))
        return super().synthesize(text, voice)


def test_callers_in_two_languages_at_once_each_hear_their_own(client: TestClient) -> None:
    services: Services = client.app.state.services  # type: ignore[attr-defined]
    voice = _RecordingVoice()
    services.tts = voice
    services.llm = _AnswersInTheirLanguage()
    said = {"en": "what is my balance", "ur": "میرا بیلنس کیا ہے"}
    turns: dict[str, dict] = {}

    def call(language: str) -> None:
        turns[language] = client.post("/v1/chat", json={"text": said[language]}).json()

    threads = [threading.Thread(target=call, args=(lang,)) for lang in ("en", "ur", "en", "ur")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    assert turns["en"]["reply"].startswith("Your balance")
    assert turns["ur"]["reply"].startswith("آپ کا بیلنس")
    # Every piece in the voice of its own caller's language.
    for text, language in voice.voices:
        assert language == ("ur" if any("؀" <= c <= "ۿ" for c in text) else "en"), text


def test_a_caller_who_hangs_up_stops_the_rest_of_the_reply(client: TestClient) -> None:
    _speaking(client, _PiecesVoice(), "One. Two. Three.")
    turn = client.post("/v1/chat", json={"text": "hello"}).json()
    services: Services = client.app.state.services  # type: ignore[attr-defined]
    speaking = services.sessions.find(turn["session_id"]).speaking  # type: ignore[union-attr]
    assert speaking is not None

    assert client.delete(f"/v1/session/{turn['session_id']}").status_code in (200, 204)

    assert speaking.wait(1, timeout=1) is None  # nothing more is made, or given out
    base = f"/v1/sessions/{turn['session_id']}/audio/{turn['audio_id']}"
    assert client.get(f"{base}/1").status_code == 404
