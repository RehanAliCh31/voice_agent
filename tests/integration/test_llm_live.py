"""Tests that talk to the real model server.

Marked `live` and deselected by default: the endpoint is on a private network,
so these cannot run in CI or on anyone else's machine. They exist so the
integration is documented and checkable by hand:

    pytest -m live

Everything these cover is also covered offline against `FakeLLMClient`. What
they add is proof that the real server still behaves the way the offline
fakes assume — the assumptions most likely to rot.
"""

from __future__ import annotations

import pytest

from voicebot.llm.openai_client import OpenAIClient

BASE_URL = "http://localhost:8001/v1"
MODEL = "qwen3.8-27b-fp8"

BALANCE_TOOL = {
    "type": "function",
    "function": {
        "name": "get_balance",
        "description": "Look up the balance of one of the customer's accounts.",
        "parameters": {
            "type": "object",
            "properties": {"account": {"type": "string", "enum": ["checking", "savings"]}},
            "required": ["account"],
        },
    },
}

pytestmark = pytest.mark.live


@pytest.fixture
def client() -> OpenAIClient:
    return OpenAIClient(base_url=BASE_URL, model=MODEL, temperature=0.0)


def test_the_server_calls_a_tool_for_an_account_question(client: OpenAIClient) -> None:
    reply = client.complete(
        [
            {"role": "system", "content": "You are a banking assistant. Use the tools provided."},
            {"role": "user", "content": "what is my savings balance?"},
        ],
        [BALANCE_TOOL],
    )
    assert reply.tool_calls, "the model answered without looking anything up"
    assert reply.tool_calls[0].name == "get_balance"
    assert reply.tool_calls[0].arguments == {"account": "savings"}


def test_the_server_does_not_call_a_tool_for_small_talk(client: OpenAIClient) -> None:
    reply = client.complete(
        [
            {"role": "system", "content": "You are a warm, concise banking assistant."},
            {"role": "user", "content": "hi there, how are you doing today?"},
        ],
        [BALANCE_TOOL],
    )
    assert reply.tool_calls == ()
    assert reply.text and len(reply.text.split()) < 40


def test_the_server_reproduces_an_exact_figure_from_a_tool_result(client: OpenAIClient) -> None:
    """The assumption the numeric guard is calibrated against.

    The model reformats what it is given -- it drops a trailing `.00` -- but it
    does not alter a real amount. If that ever stops being true, the guard's
    value-equality comparison is the thing that catches it, and this test is
    the thing that tells us why.
    """
    reply = client.complete(
        [
            {"role": "system", "content": "You are a banking assistant. Keep replies short."},
            {"role": "user", "content": "what is my savings balance?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_balance",
                            "arguments": '{"account": "savings"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"account": "savings", "balance": "$19,020.47"}',
            },
        ],
        [BALANCE_TOOL],
    )
    assert reply.text and "19,020.47" in reply.text


def test_dictated_card_digits_reach_the_tool_intact(client: OpenAIClient) -> None:
    """The one job the deleted normaliser did that nothing else does.

    v1 had `_join_digit_runs`, which glued a dictated "4 4 1 7" back into
    "4417" before the trained model saw it. Deleting that module is only safe
    if the LLM does the same thing natively. This is the proof.
    """
    freeze = {
        "type": "function",
        "function": {
            "name": "freeze_card",
            "description": "Freeze a card so nothing further goes through on it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "last4": {"type": "string", "description": "The card's last four digits."}
                },
                "required": ["last4"],
            },
        },
    }
    reply = client.complete(
        [
            {"role": "system", "content": "You are a banking assistant. Use the tools provided."},
            {"role": "user", "content": "freeze my card, it's four four one seven"},
        ],
        [freeze],
    )
    assert reply.tool_calls, "the model did not reach for the tool"
    assert reply.tool_calls[0].arguments.get("last4") == "4417"
