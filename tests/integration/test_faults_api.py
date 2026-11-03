"""Fault injection through the HTTP surface: the turn every caller actually takes.

Each test breaks one thing -- a provider, the bank, the request itself, the
caller -- and checks what the page gets: a turn with a sentence it can speak
(or a clean 4xx/5xx with a safe message when it is not a turn), never a
traceback, never a raw error, never silence, and never one caller's failure
reaching another caller.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from voicebot.api.app import create_app
from voicebot.api.audit import AuditWriter
from voicebot.api.deps import Services, check_startup
from voicebot.api.limits import ConversationLimits
from voicebot.api.routes import _turn
from voicebot.asr.base import ASREngine
from voicebot.bank.base import StorageError
from voicebot.config import AppConfig, DatabaseConfig, SessionConfig
from voicebot.errors import Cancelled
from voicebot.llm.base import FakeLLMClient, LLMClient, LLMUnavailable, Reply, ToolCall
from voicebot.messages import say
from voicebot.schemas import AudioBuffer, Transcript
from voicebot.utils import observability
from voicebot.utils.resilience import checkpoint

TRANSFER = {"amount_cents": 5_000, "payee": "sarah", "from_account": "checking"}


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app()) as test_client:
        yield test_client


def services_of(client: TestClient) -> Services:
    return client.app.state.services  # type: ignore[no-any-return]


@pytest.fixture
def logged() -> Iterator[list[logging.LogRecord]]:
    records: list[logging.LogRecord] = []

    class Keep(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = Keep()
    logging.getLogger("voicebot").addHandler(handler)
    yield records
    logging.getLogger("voicebot").removeHandler(handler)


class Faulty:
    """The bank, with the named methods failing like a lost database."""

    def __init__(self, bank: Any, *failing: str, error: Exception | None = None) -> None:
        self._bank, self._failing, self._error = bank, set(failing), error

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._bank, name)
        if name in self._failing and callable(attr):

            def broken(*args: Any, **kwargs: Any) -> Any:
                raise self._error or StorageError(f"{name}: connection lost")

            return broken
        return attr


class Down(LLMClient):
    def complete(self, messages: Any, tools: Any, tool_choice: str = "auto") -> Reply:
        raise LLMUnavailable("the model is down")


# --- one caller's turn must not hold up, or break, another's ------------------------


def test_a_slow_turn_does_not_freeze_the_server(client: TestClient) -> None:
    class Slow(LLMClient):
        def complete(self, messages: Any, tools: Any, tool_choice: str = "auto") -> Reply:
            time.sleep(1.0)
            return Reply(text="Hi.")

    services_of(client).llm = Slow()
    turn = threading.Thread(target=lambda: client.post("/v1/chat", json={"text": "hello"}))
    turn.start()
    time.sleep(0.2)
    started = time.monotonic()
    assert client.get("/health").status_code == 200
    assert time.monotonic() - started < 0.5, "health waited behind someone else's turn"
    turn.join(5)


def test_one_caller_failing_leaves_another_unaffected(client: TestClient) -> None:
    class FailsForOne(LLMClient):
        def complete(self, messages: Any, tools: Any, tool_choice: str = "auto") -> Reply:
            said = str(messages[-1].get("content", ""))
            time.sleep(0.2)
            if "boom" in said:
                raise RuntimeError("a bug that only this caller hits")
            return Reply(text="Fine, thanks.")

    services_of(client).llm = FailsForOne()
    answers: dict[str, Any] = {}

    def ask(session: str, text: str) -> None:
        answers[session] = client.post("/v1/chat", json={"text": text, "session_id": session})

    threads = [
        threading.Thread(target=ask, args=("caller-a", "boom")),
        threading.Thread(target=ask, args=("caller-b", "how are you?")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    failed, fine = answers["caller-a"], answers["caller-b"]
    assert failed.status_code == 200 and failed.json()["fallback"] == "internal"
    assert failed.json()["reply"] == say("internal", "en")
    assert fine.status_code == 200 and fine.json()["reply"] == "Fine, thanks."


def test_yes_pressed_twice_at_once_moves_the_money_once(client: TestClient) -> None:
    services = services_of(client)
    session = services.sessions.get("double")
    session.conversation.append_user("send sarah fifty")
    session.executor.execute(ToolCall("c0", "transfer_money", TRANSFER))
    session.executor.pin_accepted()
    services.llm = FakeLLMClient([Reply(text="Done.")] * 4)
    before = services.bank.balance("checking")

    answers: list[Any] = []

    def press() -> None:
        answers.append(client.post("/v1/confirm", json={"session_id": "double"}))

    threads = [threading.Thread(target=press) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    assert services.bank.balance("checking") == before - 5_000
    assert all(answer.status_code == 200 for answer in answers)


# --- the bank failing around a turn that has already happened --------------------------


def test_a_turn_record_the_database_refuses_is_kept_not_lost(
    client: TestClient, tmp_path: Path
) -> None:
    services = services_of(client)
    dead = tmp_path / "dead.jsonl"
    services.audit = AuditWriter(
        Faulty(services.bank, "log_turn"), dead, retries=1, sleep=lambda _: None  # type: ignore[arg-type]
    )
    services.llm = FakeLLMClient([Reply(text="Hello there.")])

    response = client.post("/v1/chat", json={"text": "hello", "session_id": "audit-1"})

    assert response.status_code == 200 and response.json()["reply"] == "Hello there."
    assert services.audit.flush()
    (line,) = dead.read_text(encoding="utf-8").splitlines()
    kept = json.loads(line)
    assert kept["session_id"] == "audit-1" and kept["bot_text"] == "Hello there."
    assert observability.event_counts()["audit_dead_letter"] == 1
    assert "voicebot_audit_dead_letters_total 1" in client.get("/metrics").text


def test_a_ledger_that_cannot_be_read_does_not_undo_the_turn(client: TestClient) -> None:
    services = services_of(client)
    services.llm = FakeLLMClient([Reply(text="Hello there.")])
    services.bank = Faulty(services.bank, "accounts", "bills")  # type: ignore[assignment]

    response = client.post("/v1/chat", json={"text": "hello"})

    assert response.status_code == 200
    assert response.json()["accounts"] is None and response.json()["bills"] is None


# --- whatever breaks, the page gets a sentence or a safe message -------------------------


def test_a_crash_inside_the_turn_is_a_spoken_sentence_with_its_call_id(
    client: TestClient, logged: list[logging.LogRecord]
) -> None:
    class Broken(LLMClient):
        def complete(self, messages: Any, tools: Any, tool_choice: str = "auto") -> Reply:
            raise RuntimeError("an engine bug")

    services_of(client).llm = Broken()
    response = client.post("/v1/chat", json={"text": "hello", "session_id": "crash-1"})

    body = response.json()
    assert response.status_code == 200
    assert body["reply"] == say("internal", "en") and body["call_id"] == "crash-1"
    assert "RuntimeError" not in body["reply"] and "Traceback" not in response.text
    assert any(
        getattr(r, "call_id", None) == "crash-1" or "call=crash-1" in r.getMessage()
        for r in logged
        if r.levelno >= logging.ERROR
    ), "the failure is logged with the call it happened in"


def test_a_malformed_request_never_echoes_what_was_sent(client: TestClient) -> None:
    response = client.post("/v1/cards/4417/pin", json={"current_pin": "1234"})
    assert response.status_code == 422
    assert "1234" not in response.text
    assert response.json()["fields"] == ["new_pin"]


def test_an_unexpected_error_outside_a_turn_is_a_safe_500() -> None:
    # As a browser meets it: the TestClient would otherwise re-raise the error.
    with TestClient(create_app(), raise_server_exceptions=False) as client:
        services = services_of(client)
        services.bank = Faulty(services.bank, "cards", error=TypeError("NoneType has no len"))  # type: ignore[assignment]
        response = client.get("/v1/cards")

    assert response.status_code == 500
    body = response.json()
    assert body["code"] == "internal" and "NoneType" not in response.text
    assert body["detail"] == say("internal", "en")


def test_the_bank_down_outside_a_turn_is_a_503(client: TestClient) -> None:
    services = services_of(client)
    services.bank = Faulty(services.bank, "accounts", "bills")  # type: ignore[assignment]
    response = client.get("/v1/accounts")
    assert response.status_code == 503
    assert response.json()["code"] == "database_error"
    assert "connection lost" not in response.text


# --- the conversation's own limits ------------------------------------------------------


def test_repeated_failures_hand_the_conversation_to_a_person(client: TestClient) -> None:
    services = services_of(client)
    services.llm = Down()
    replies = [
        client.post("/v1/chat", json={"text": f"hello {n}", "session_id": "stuck"}).json()
        for n in range(3)
    ]

    assert [r["fallback"] for r in replies[:2]] == ["unavailable", "unavailable"]
    last = replies[2]
    assert last["fallback"] == "escalated" and last["handed_off"]
    assert last["handed_off"] in last["reply"]
    assert len(services.bank.handoffs()) == 1
    assert observability.event_counts()["escalations"] == 1


def test_no_case_can_be_opened_either_gives_the_helpline(client: TestClient) -> None:
    services = services_of(client)
    services.llm = Down()
    # The store opens each conversation's executor on its own bank reference.
    faulty = Faulty(services.bank, "open_handoff")
    services.bank = services.sessions.bank = faulty  # type: ignore[assignment,attr-defined]
    for session in list(range(3)):
        last = client.post("/v1/chat", json={"text": f"hi {session}", "session_id": "stuck2"})
    assert last.json()["reply"] == say("cannot_complete", "en")


def test_a_conversation_is_warned_then_closed_and_starts_afresh(client: TestClient) -> None:
    services = services_of(client)
    services.limits = ConversationLimits(SessionConfig(max_turns=3, warn_turns=1))
    services.llm = FakeLLMClient([Reply(text=f"Reply {n}.") for n in range(10)])

    replies = [
        client.post("/v1/chat", json={"text": f"q{n}", "session_id": "long"}).json()["reply"]
        for n in range(4)
    ]

    assert replies[0] == "Reply 0."
    assert replies[1].endswith(say("session_warning", "en"))
    assert replies[3] == say("session_ended", "en")
    # A fresh conversation: the closed one's history is gone.
    session = services.sessions.get("long")
    assert [m for m in session.conversation.messages if m.get("role") == "user"] == []


# --- health, shutdown and startup -------------------------------------------------------


def test_ready_says_not_ready_when_the_bank_is_down(client: TestClient) -> None:
    services = services_of(client)
    services.bank = Faulty(services.bank, "ping")  # type: ignore[assignment]
    services._ready = (0.0, False, "")
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["checks"]["database"].startswith("unreachable")


def test_a_draining_service_says_it_is_restarting(client: TestClient) -> None:
    services = services_of(client)
    services.draining = True
    try:
        response = client.post("/v1/chat", json={"text": "hello"})
        assert response.json()["reply"] == say("restarting", "en")
        assert client.get("/ready").status_code == 503
    finally:
        services.draining = False


def test_shutdown_closes_every_part_even_when_one_fails(client: TestClient) -> None:
    services = services_of(client)
    closed: list[str] = []

    class Breaks(ASREngine):
        def transcribe(self, audio: AudioBuffer, language: str | None = None) -> Transcript:
            raise AssertionError

        def close(self) -> None:
            raise RuntimeError("would not close")

    services.asr = Breaks()
    original = services.bank.close
    services.bank.close = lambda: closed.append("bank") or original()  # type: ignore[method-assign]
    services.drain(timeout_s=0.1)
    services.close()
    assert closed == ["bank"], "the bank was still closed after the recognisers failed to"


def test_startup_refuses_a_deployment_missing_what_it_needs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VOICEBOT_DB_USER", raising=False)
    config = AppConfig(database=DatabaseConfig(engine="postgres"))
    with pytest.raises(RuntimeError, match="VOICEBOT_DB_USER"):
        check_startup(config)


# --- the caller leaving mid-turn ------------------------------------------------------------


class _Leaves:
    """A request whose caller hangs up after the turn has started."""

    def __init__(self, after_s: float) -> None:
        self.gone_at = time.monotonic() + after_s

    async def is_disconnected(self) -> bool:
        return time.monotonic() >= self.gone_at


def test_a_caller_who_hangs_up_mid_turn_stops_the_work(client: TestClient) -> None:
    services = services_of(client)
    reached_the_voice: list[bool] = []

    def work(session, pipeline):  # type: ignore[no-untyped-def]
        time.sleep(0.6)  # the model, say
        checkpoint("speaking")
        reached_the_voice.append(True)
        return pipeline.say("never heard")

    response = asyncio.run(_turn(_Leaves(0.2), services, "leaving", work))  # type: ignore[arg-type]

    assert response.status_code == 499
    assert reached_the_voice == []
    session = services.sessions.get("leaving")
    assert session.lock.acquire(blocking=False), "the conversation was released"
    session.lock.release()
    assert services.in_flight == 0
    assert observability.event_counts()["turns_abandoned"] == 1


def test_cancelled_is_never_turned_into_a_spoken_fallback(client: TestClient) -> None:
    services = services_of(client)

    def work(session, pipeline):  # type: ignore[no-untyped-def]
        raise Cancelled("the caller left before the model")

    with pytest.raises(Cancelled):
        services.take_turn("gone", work)


# --- nothing left behind after failed calls ------------------------------------------------


def test_failed_calls_leave_no_threads_or_locks_behind(client: TestClient) -> None:
    services = services_of(client)
    services.llm = Down()
    client.post("/v1/chat", json={"text": "warm up", "session_id": "leak-0"})
    before = threading.active_count()

    for n in range(10):
        client.post("/v1/chat", json={"text": "hello", "session_id": f"leak-{n % 3}"})

    assert threading.active_count() <= before + 1
    assert services.in_flight == 0
    for n in range(3):
        session = services.sessions.get(f"leak-{n}")
        assert session.lock.acquire(blocking=False)
        session.lock.release()


def test_the_page_gets_the_same_fallback_sentences_as_the_bot(client: TestClient) -> None:
    table = client.get("/v1/messages").json()
    assert table["connection_lost"]["ur"] and table["timed_out"]["en"]
    assert table["internal"]["en"] == say("internal", "en")


def test_turn_records_go_wherever_the_service_reads_the_bank_from(client: TestClient) -> None:
    # Found under PostgreSQL: the writer kept the bank it was built with, so
    # once services.bank was replaced the records went to the old one.
    from voicebot.bank.sqlite_bank import SqliteBank

    services = services_of(client)
    replacement = SqliteBank()
    services.bank = replacement
    services.llm = FakeLLMClient([Reply(text="Hello there.")])

    client.post("/v1/chat", json={"text": "hello", "session_id": "swapped"})

    assert services.audit is not None and services.audit.flush()
    rows = replacement._db.execute("SELECT session_id FROM turns").fetchall()
    assert [row["session_id"] for row in rows] == ["swapped"]
