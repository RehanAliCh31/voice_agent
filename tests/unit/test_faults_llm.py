"""Fault injection: the model's endpoint failing every way it has been seen to fail.

`_post` is the client's one seam to the network, so each test makes it time
out, rate-limit, error, drop the connection or answer nonsense, and checks
that what comes out is either a usable Reply or an LLMUnavailable of the
right kind -- never another exception, never an unbounded wait.
"""

from __future__ import annotations

import email.message
import http.client
import io
import logging
import urllib.error
from collections.abc import Iterator
from typing import Any

import pytest

from voicebot.errors import ErrorCode, call_context
from voicebot.llm import openai_client
from voicebot.llm.base import ContextOverflow, FakeLLMClient, LLMUnavailable, Reply
from voicebot.llm.openai_client import FallbackLLMClient, OpenAIClient
from voicebot.utils.resilience import BreakerState, breaker, deadline_context


def _http(code: int, body: str = "", retry_after: str | None = None) -> urllib.error.HTTPError:
    headers = email.message.Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError("http://x/v1", code, "error", headers, io.BytesIO(body.encode()))


def _ok(content: str = "Hello.", finish: str = "stop") -> dict[str, Any]:
    return {"choices": [{"message": {"content": content}, "finish_reason": finish}]}


class Server:
    """Answers each request with the next of `script`: a body, or an exception to raise."""

    def __init__(self, *script: Any) -> None:
        self.script = list(script)
        self.timeouts: list[float] = []

    def __call__(self, url: str, payload: dict, headers: dict, timeout: float) -> Any:
        self.timeouts.append(timeout)
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, BaseException):
            raise item
        return item


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    slept: list[float] = []
    monkeypatch.setattr("voicebot.utils.resilience.time.sleep", slept.append)
    return slept


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


def _client(monkeypatch: pytest.MonkeyPatch, server: Server, **kwargs: Any) -> OpenAIClient:
    monkeypatch.setattr(openai_client, "_post", server)
    return OpenAIClient(base_url="http://x/v1", model="m", **kwargs)


def test_a_timeout_is_retried_once_then_reported_as_a_timeout(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    server = Server(TimeoutError("read timed out"))
    with pytest.raises(LLMUnavailable) as raised:
        _client(monkeypatch, server, max_retries=1).complete([], [])
    assert raised.value.code is ErrorCode.PROVIDER_TIMEOUT
    assert len(server.timeouts) == 2


def test_a_429_waits_as_long_as_it_is_told_then_succeeds(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    server = Server(_http(429, retry_after="1"), _ok("Fine."))
    with deadline_context(10):
        reply = _client(monkeypatch, server).complete([], [])
    assert reply.text == "Fine."
    assert no_sleep and no_sleep[0] >= 1.0


def test_a_429_asking_for_longer_than_the_caller_can_wait_is_not_retried(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    server = Server(_http(429, retry_after="60"), _ok())
    with deadline_context(5), pytest.raises(LLMUnavailable) as raised:
        _client(monkeypatch, server).complete([], [])
    assert raised.value.code is ErrorCode.RATE_LIMITED
    assert len(server.timeouts) == 1 and no_sleep == []


def test_a_500_is_retried_and_the_second_answer_used(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    server = Server(_http(500, "boom"), _ok("Recovered."))
    assert _client(monkeypatch, server).complete([], []).text == "Recovered."


def test_a_400_is_never_retried(monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]) -> None:
    server = Server(_http(400, "bad request"), _ok())
    with pytest.raises(LLMUnavailable) as raised:
        _client(monkeypatch, server, max_retries=2).complete([], [])
    assert raised.value.code is ErrorCode.INVALID_INPUT
    assert len(server.timeouts) == 1


def test_a_refused_key_is_an_auth_failure_and_not_retried(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    server = Server(_http(401))
    with pytest.raises(LLMUnavailable) as raised:
        _client(monkeypatch, server, max_retries=2).complete([], [])
    assert raised.value.code is ErrorCode.AUTH_FAILURE and len(server.timeouts) == 1


def test_a_conversation_too_long_for_the_model_is_named_as_such(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    server = Server(_http(400, "This model's maximum context length is 32768 tokens"))
    with pytest.raises(ContextOverflow):
        _client(monkeypatch, server).complete([], [])


@pytest.mark.parametrize(
    "dropped",
    [
        http.client.IncompleteRead(b'{"choi'),
        http.client.RemoteDisconnected("closed"),
        ConnectionResetError("reset by peer"),
        urllib.error.URLError("connection refused"),
    ],
)
def test_a_connection_dropped_mid_answer_is_retryable(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float], dropped: Exception
) -> None:
    server = Server(dropped, _ok("Back."))
    assert _client(monkeypatch, server).complete([], []).text == "Back."


@pytest.mark.parametrize(
    "garbage",
    [
        ["not", "an", "object"],
        {"choices": [None]},
        {"choices": "nope"},
        {"error": {"message": "the engine is overloaded"}},
    ],
)
def test_a_malformed_answer_is_an_llm_failure_not_a_crash(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float], garbage: Any
) -> None:
    server = Server(garbage)
    with pytest.raises(LLMUnavailable):
        _client(monkeypatch, server).complete([], [])


def test_an_answer_that_is_not_json_is_retried(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    server = Server(ValueError("Expecting value"), _ok("Fine."))
    assert _client(monkeypatch, server).complete([], []).text == "Fine."


def test_an_empty_answer_is_an_empty_reply_for_the_loop_to_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _client(monkeypatch, Server({"choices": []})).complete([], []) == Reply()


def test_content_sent_as_parts_is_read_as_text(monkeypatch: pytest.MonkeyPatch) -> None:
    body = {"choices": [{"message": {"content": [{"type": "text", "text": "Hi there."}]}}]}
    assert _client(monkeypatch, Server(body)).complete([], []).text == "Hi there."


def test_a_reply_cut_off_by_the_token_limit_ends_at_its_last_whole_sentence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = _ok("Your balance is $19,020.00. Your bills are due on the", finish="length")
    assert _client(monkeypatch, Server(body)).complete([], []).text == (
        "Your balance is $19,020.00."
    )


def test_broken_tool_call_json_becomes_an_empty_argument_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "get_balance", "arguments": '{"acc'}},
                        "not a call",
                    ]
                }
            }
        ]
    }
    reply = _client(monkeypatch, Server(body)).complete([], [])
    assert [(c.name, c.arguments) for c in reply.tool_calls] == [("get_balance", {})]


def test_each_attempt_is_cut_to_the_turns_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    server = Server(_ok())
    with deadline_context(3):
        _client(monkeypatch, server, timeout_s=12).complete([], [])
    assert server.timeouts[0] <= 3.0


def test_a_failing_model_is_skipped_once_its_circuit_opens(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float], logged: list[logging.LogRecord]
) -> None:
    server = Server(_http(503))
    client = _client(monkeypatch, server, max_retries=0)
    for _ in range(breaker("llm").failures):
        with pytest.raises(LLMUnavailable):
            client.complete([], [])
    assert breaker("llm").state is BreakerState.OPEN

    calls = len(server.timeouts)
    with call_context("call-3"), pytest.raises(LLMUnavailable, match="circuit is open"):
        client.complete([], [])
    assert len(server.timeouts) == calls, "an open circuit does not call the server"
    assert any(r.levelno == logging.CRITICAL for r in logged), "opening it raised an alert"


def test_the_backup_model_answers_when_the_main_one_is_down() -> None:
    class Down(FakeLLMClient):
        def complete(self, messages, tools, tool_choice="auto"):  # type: ignore[no-untyped-def]
            raise LLMUnavailable("down", code=ErrorCode.PROVIDER_TIMEOUT)

    backup = FakeLLMClient([Reply(text="From the backup.")])
    client = FallbackLLMClient(Down([]), backup)
    assert client.complete([{"role": "user", "content": "hi"}], []).text == "From the backup."


def test_a_request_the_main_model_refused_is_not_sent_to_the_backup() -> None:
    class Refuses(FakeLLMClient):
        def complete(self, messages, tools, tool_choice="auto"):  # type: ignore[no-untyped-def]
            raise LLMUnavailable("bad", code=ErrorCode.INVALID_INPUT)

    backup = FakeLLMClient([])
    with pytest.raises(LLMUnavailable):
        FallbackLLMClient(Refuses([]), backup).complete([], [])
    assert backup.calls == []


def test_an_amount_the_limit_cut_in_half_is_not_taken_for_a_full_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Measured: an Urdu list of bills cut off at "$1,200." of "$1,200.00".
    body = _ok("Rent is $1,200.00. Your checking holds $2,483.", finish="length")
    assert _client(monkeypatch, Server(body)).complete([], []).text == "Rent is $1,200.00."


def test_the_model_is_asked_not_to_think_before_answering(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[dict] = []

    def server(url: str, payload: dict, headers: dict, timeout: float) -> dict:
        sent.append(payload)
        return _ok()

    monkeypatch.setattr(openai_client, "_post", server)
    OpenAIClient(base_url="http://x/v1", model="m").complete([], [])
    OpenAIClient(base_url="http://x/v1", model="m", disable_thinking=False).complete([], [])
    assert sent[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "chat_template_kwargs" not in sent[1]
