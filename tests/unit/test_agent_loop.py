"""The agent loop: the LLM, its tools, and the guard, in one turn.

Every test here runs against `FakeLLMClient`, so the whole loop — tool
calling, the confirmation gate, the guard, the failure paths — is exercised
with no endpoint and no network.
"""

from __future__ import annotations

from typing import Any

import pytest

from voicebot.agent.executor import ToolExecutor
from voicebot.agent.loop import Agent, Conversation
from voicebot.bank.sqlite_bank import SqliteBank
from voicebot.llm.base import FakeLLMClient, LLMClient, LLMUnavailable, Reply, ToolCall
from voicebot.llm.prompts import SYSTEM_PROMPT, UNAVAILABLE

TRANSFER = {"amount_cents": 5_000, "payee": "sarah", "from_account": "checking"}


@pytest.fixture
def bank() -> SqliteBank:
    return SqliteBank()


@pytest.fixture
def executor(bank: SqliteBank) -> ToolExecutor:
    return ToolExecutor(bank)


def agent_with(replies: list[Reply], executor: ToolExecutor, **kwargs: Any) -> Agent:
    return Agent(FakeLLMClient(replies), executor, **kwargs)


# --- the simple path ---------------------------------------------------------


def test_a_social_turn_calls_no_tools(executor: ToolExecutor) -> None:
    agent = agent_with([Reply(text="I'm well, thanks. How can I help?")], executor)
    out = agent.handle("how are you?", Conversation("s1"))

    assert out.text == "I'm well, thanks. How can I help?"
    assert out.tool_calls == ()
    assert out.rounds == 1
    assert out.guard_hit is False


def test_the_system_prompt_is_sent_once_and_first(executor: ToolExecutor) -> None:
    client = FakeLLMClient([Reply(text="one"), Reply(text="two")])
    agent = Agent(client, executor)
    conversation = Conversation("s1")

    agent.handle("hello", conversation)
    agent.handle("again", conversation)

    sent = client.calls[1][0]
    assert sent[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert sum(1 for m in sent if m["role"] == "system") == 1


def test_the_tools_are_offered_on_every_call(executor: ToolExecutor) -> None:
    client = FakeLLMClient([Reply(text="hi")])
    Agent(client, executor).handle("hello", Conversation("s1"))
    assert len(client.calls[0][1]) == 9


# --- tool calling ------------------------------------------------------------


def test_a_tool_call_is_executed_and_the_result_fed_back(executor: ToolExecutor) -> None:
    client = FakeLLMClient(
        [
            Reply(tool_calls=(ToolCall("c1", "get_balance", {"account": "savings"}),)),
            Reply(text="Your savings balance is $19,020.00."),
        ]
    )
    out = Agent(client, executor).handle("balance?", Conversation("s1"))

    assert out.text == "Your savings balance is $19,020.00."
    assert out.rounds == 2
    assert out.guard_hit is False
    assert out.tool_calls[0].name == "get_balance"
    assert client.calls[1][0][-1]["role"] == "tool"


def test_several_tool_calls_in_one_reply_all_run(executor: ToolExecutor) -> None:
    client = FakeLLMClient(
        [
            Reply(
                tool_calls=(
                    ToolCall("c1", "get_balance", {"account": "savings"}),
                    ToolCall("c2", "get_balance", {"account": "checking"}),
                )
            ),
            Reply(text="Savings $19,020.00, checking $2,483.50."),
        ]
    )
    out = Agent(client, executor).handle("both balances?", Conversation("s1"))

    assert len(out.tool_results) == 2
    assert out.guard_hit is False


def test_max_tool_rounds_is_enforced(executor: ToolExecutor) -> None:
    forever = Reply(tool_calls=(ToolCall("c", "list_bills", {}),))
    agent = agent_with([forever] * 10, executor, max_tool_rounds=3)
    out = agent.handle("x", Conversation("s1"))

    assert out.rounds == 3
    assert out.text.strip()


def test_a_tool_error_reaches_the_model_as_json(executor: ToolExecutor) -> None:
    client = FakeLLMClient(
        [
            Reply(tool_calls=(ToolCall("c1", "get_spending", {"period": "october"}),)),
            Reply(text="I can do this week or last month — which would you like?"),
        ]
    )
    Agent(client, executor).handle("spending in october?", Conversation("s1"))

    tool_message = client.calls[1][0][-1]
    assert tool_message["role"] == "tool"
    assert "error" in tool_message["content"]


# --- failure paths -----------------------------------------------------------


def test_an_empty_reply_becomes_a_sayable_sentence(executor: ToolExecutor) -> None:
    out = agent_with([Reply(text=None)], executor).handle("x", Conversation("s1"))
    assert out.text.strip()


def test_an_unreachable_endpoint_is_spoken_not_raised(executor: ToolExecutor) -> None:
    class Down(LLMClient):
        def complete(self, messages: Any, tools: Any) -> Reply:
            raise LLMUnavailable("down")

    out = Agent(Down(), executor).handle("x", Conversation("s1"))
    assert out.text == UNAVAILABLE


# --- the guard inside the loop -----------------------------------------------


def test_an_invented_figure_is_replaced_before_it_is_returned(executor: ToolExecutor) -> None:
    client = FakeLLMClient(
        [
            Reply(tool_calls=(ToolCall("c1", "get_balance", {"account": "savings"}),)),
            Reply(text="Your savings balance is $31,415.00."),
        ]
    )
    out = Agent(client, executor).handle("balance?", Conversation("s1"))

    assert out.guard_hit is True
    assert "31,415" not in out.text
    assert "19,020.00" in out.text  # the fallback carries the real figure


def test_the_fallback_is_generic_when_no_tool_returned_money(executor: ToolExecutor) -> None:
    out = agent_with([Reply(text="Your balance is $500.00.")], executor).handle(
        "x", Conversation("s1")
    )
    assert out.guard_hit is True
    assert "500" not in out.text


# --- confirmation ------------------------------------------------------------


def test_a_confirming_tool_does_not_execute_inside_the_loop(
    bank: SqliteBank, executor: ToolExecutor
) -> None:
    client = FakeLLMClient(
        [
            Reply(tool_calls=(ToolCall("c1", "transfer_money", TRANSFER),)),
            Reply(text="Transfer $50.00 to sarah from checking. Shall I go ahead?"),
        ]
    )
    out = Agent(client, executor).handle("send sarah fifty", Conversation("s1"))

    assert bank.balance("checking") == 248_350
    assert executor.pending is not None
    assert out.guard_hit is False  # $50.00 came from the summary, which is a tool result


def test_resolve_pending_executes_and_tells_the_model_it_happened(
    bank: SqliteBank, executor: ToolExecutor
) -> None:
    executor.execute(ToolCall("c1", "transfer_money", TRANSFER))
    conversation = Conversation("s1")
    client = FakeLLMClient([Reply(text="Done. $50.00 sent to sarah.")])

    out = Agent(client, executor).resolve_pending(True, conversation)

    assert bank.balance("checking") == 243_350
    # The model's next turn must know the money moved, or "did that go
    # through?" gets answered from a history that still says it is waiting.
    assert any(m["role"] == "tool" for m in conversation.messages)
    assert out.text


def test_resolve_pending_cancel_leaves_the_money_alone(
    bank: SqliteBank, executor: ToolExecutor
) -> None:
    executor.execute(ToolCall("c1", "transfer_money", TRANSFER))
    client = FakeLLMClient([Reply(text="No problem, I've cancelled that.")])

    Agent(client, executor).resolve_pending(False, Conversation("s1"))

    assert bank.balance("checking") == 248_350
    assert executor.pending is None


def test_resolve_pending_with_nothing_pending_still_answers(executor: ToolExecutor) -> None:
    client = FakeLLMClient([Reply(text="There wasn't anything waiting.")])
    out = Agent(client, executor).resolve_pending(True, Conversation("s1"))
    assert out.text.strip()


# --- conversation ------------------------------------------------------------


def test_the_turn_index_advances(executor: ToolExecutor) -> None:
    conversation = Conversation("s1")
    agent = agent_with([Reply(text="a"), Reply(text="b")], executor)
    agent.handle("one", conversation)
    agent.handle("two", conversation)
    assert conversation.turn_index == 2


def test_trim_keeps_the_system_prompt_and_bounds_the_history() -> None:
    conversation = Conversation("s1")
    conversation.messages.append({"role": "system", "content": SYSTEM_PROMPT})
    for i in range(60):
        conversation.append_user(f"turn {i}")

    conversation.trim(max_turns=20)

    assert conversation.messages[0]["role"] == "system"
    assert len(conversation.messages) <= 21
    assert conversation.messages[-1]["content"] == "turn 59"
