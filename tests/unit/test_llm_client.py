"""The LLM client: parsing replies, retrying, and failing in a sayable way.

Nothing here touches the network. `_parse` is a static method precisely so the
shape of a vLLM response can be tested without one, and `_post` is a module
level function so a test can replace the single seam that does the I/O.
"""

from __future__ import annotations

from typing import Any

import pytest

from voicebot.llm import openai_client
from voicebot.llm.base import FakeLLMClient, LLMUnavailable, Reply, ToolCall
from voicebot.llm.openai_client import OpenAIClient, strip_thinking


def _message(**message: Any) -> dict[str, Any]:
    return {"choices": [{"message": message}]}


# --- parsing -----------------------------------------------------------------


def test_a_text_reply_with_tool_calls_absent_is_parsed() -> None:
    # vLLM omits the key entirely on a text reply rather than sending null,
    # so anything reading message["tool_calls"] raises KeyError in production.
    reply = OpenAIClient._parse(_message(role="assistant", content="Hello."))
    assert reply.text == "Hello."
    assert reply.tool_calls == ()


def test_a_tool_call_reply_is_parsed_with_arguments_decoded() -> None:
    body = _message(
        content=None,
        tool_calls=[
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "get_balance", "arguments": '{"account": "savings"}'},
            }
        ],
    )
    reply = OpenAIClient._parse(body)
    assert reply.text is None
    assert reply.tool_calls == (
        ToolCall(id="c1", name="get_balance", arguments={"account": "savings"}),
    )


def test_malformed_tool_arguments_do_not_raise() -> None:
    # A model that emits broken JSON should produce an empty argument dict the
    # executor can reject politely, not a 500.
    body = _message(
        content=None,
        tool_calls=[
            {"id": "c1", "type": "function", "function": {"name": "get_balance", "arguments": "{"}}
        ],
    )
    assert OpenAIClient._parse(body).tool_calls[0].arguments == {}


def test_arguments_that_are_already_a_dict_are_accepted() -> None:
    body = _message(
        content=None,
        tool_calls=[
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "get_balance", "arguments": {"account": "savings"}},
            }
        ],
    )
    assert OpenAIClient._parse(body).tool_calls[0].arguments == {"account": "savings"}


def test_a_reply_with_no_choices_is_empty_rather_than_an_error() -> None:
    assert OpenAIClient._parse({"choices": []}) == Reply(text=None, tool_calls=())


# --- thinking ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("<think>hmm</think>Your balance is $5.", "Your balance is $5."),
        ("<think>\nmulti\nline\n</think>  Hello.", "Hello."),
        # What qwen3.6-35b-a3b-fp8 sends after a tool call, now and then: the
        # server's template opens the block itself, so only the closing tag
        # arrives, and without this the answer was read aloud twice with the
        # tag between.
        (
            "Enter your PIN to approve the transfer.\n</think>\n\n"
            "Enter your PIN to approve the transfer.",
            "Enter your PIN to approve the transfer.",
        ),
        ("plain text", "plain text"),
        ("", ""),
    ],
)
def test_a_thinking_block_is_stripped(raw: str, expected: str) -> None:
    assert strip_thinking(raw) == expected


def test_parsing_strips_thinking_from_the_content() -> None:
    reply = OpenAIClient._parse(_message(content="<think>reasoning</think>Hello."))
    assert reply.text == "Hello."


# --- transport ---------------------------------------------------------------


class _Recorder:
    """Stands in for the one function that does I/O."""

    def __init__(self, error: Exception | None = None, body: dict | None = None) -> None:
        self.error = error
        self.body = body
        self.calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        self.calls += 1
        if self.error:
            raise self.error
        assert self.body is not None
        return self.body


def test_the_client_retries_then_raises_llm_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    post = _Recorder(error=OSError("connection refused"))
    monkeypatch.setattr(openai_client, "_post", post)
    client = OpenAIClient(base_url="http://x/v1", model="m", max_retries=2, backoff_s=0.0)

    with pytest.raises(LLMUnavailable):
        client.complete([], [])

    assert post.calls == 3  # the first attempt plus two retries


def test_a_successful_call_does_not_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    post = _Recorder(body=_message(content="Hello."))
    monkeypatch.setattr(openai_client, "_post", post)
    client = OpenAIClient(base_url="http://x/v1", model="m", backoff_s=0.0)

    assert client.complete([], []).text == "Hello."
    assert post.calls == 1


def test_the_request_names_the_model_and_carries_the_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def capture(url: str, payload: dict, headers: dict, timeout: float) -> dict:
        captured.update(url=url, payload=payload)
        return _message(content="ok")

    monkeypatch.setattr(openai_client, "_post", capture)
    tools = [{"type": "function", "function": {"name": "get_balance"}}]
    OpenAIClient(base_url="http://x/v1", model="qwen3.8-27b-fp8").complete([], tools)

    assert captured["url"] == "http://x/v1/chat/completions"
    assert captured["payload"]["model"] == "qwen3.8-27b-fp8"
    assert captured["payload"]["tools"] == tools


def test_tools_are_omitted_when_there_are_none(monkeypatch: pytest.MonkeyPatch) -> None:
    # Sending "tools": [] makes some servers refuse the request outright.
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        openai_client,
        "_post",
        lambda url, payload, headers, timeout: (
            captured.update(payload=payload) or _message(content="ok")
        ),
    )
    OpenAIClient(base_url="http://x/v1", model="m").complete([], [])
    assert "tools" not in captured["payload"]


# --- the fake ----------------------------------------------------------------


def test_the_fake_returns_its_script_in_order_and_records_calls() -> None:
    fake = FakeLLMClient([Reply(text="one"), Reply(text="two")])

    assert fake.complete([{"role": "user", "content": "a"}], []).text == "one"
    assert fake.complete([], []).text == "two"
    assert len(fake.calls) == 2
    assert fake.calls[0][0] == [{"role": "user", "content": "a"}]


def test_the_fake_raises_when_its_script_runs_out() -> None:
    # A test that over-runs its script is a test bug; it should say so loudly
    # rather than quietly returning something plausible.
    fake = FakeLLMClient([Reply(text="only one")])
    fake.complete([], [])
    with pytest.raises(IndexError):
        fake.complete([], [])


# --- the fake refuses what the real server refuses ----------------------------


def test_the_fake_refuses_a_history_with_nobody_in_it() -> None:
    """A tool call and its result, and nothing a customer ever said.

    The real server answers that with 400 "no user query found", and the fake
    used to answer it happily -- so a bug that broke every screen action in
    the browser passed the whole suite. It raises now, for the same reason it
    raises when its script runs out.
    """
    fake = FakeLLMClient([Reply(text="sure")])

    with pytest.raises(ValueError, match="user"):
        fake.complete(
            [
                {"role": "system", "content": "..."},
                {"role": "assistant", "tool_calls": [{"id": "c1"}]},
                {"role": "tool", "tool_call_id": "c1", "content": "{}"},
            ],
            [],
        )


def test_the_fake_still_allows_an_empty_history() -> None:
    """Client tests call it with nothing at all, which is not the same bug."""
    assert FakeLLMClient([Reply(text="hello")]).complete([], []).text == "hello"
