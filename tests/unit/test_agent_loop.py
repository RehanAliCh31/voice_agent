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


# --- trimming must not split a tool call from its result --------------------


def test_trim_never_orphans_a_tool_message(executor: ToolExecutor) -> None:
    """A `tool` message whose assistant parent was dropped is a 400.

    The client treats a 4xx as non-retryable, so the customer would hear the
    outage apology on every turn until the orphan scrolled out of the window.
    """
    conversation = Conversation("s1")
    agent = Agent(
        FakeLLMClient(
            [
                (
                    Reply(tool_calls=(ToolCall(f"c{i}", "list_bills", {}),))
                    if i % 2
                    else Reply(text=f"reply {i}")
                )
                for i in range(40)
            ]
            + [Reply(text="done")] * 40
        ),
        executor,
        max_turns=3,
    )
    for i in range(12):
        agent.handle(f"turn {i}", conversation)

    roles = [m["role"] for m in conversation.messages]
    for index, role in enumerate(roles):
        if role != "tool":
            continue
        earlier = roles[:index]
        assert "assistant" in earlier, f"orphaned tool message at {index}: {roles}"


def test_trim_counts_turns_not_messages(executor: ToolExecutor) -> None:
    # A tool turn is several messages. Counting messages would make the bot
    # forget roughly five exchanges instead of twenty.
    conversation = Conversation("s1")
    agent = Agent(FakeLLMClient([Reply(text="ok")] * 40), executor, max_turns=5)
    for i in range(12):
        agent.handle(f"turn {i}", conversation)

    users = [m for m in conversation.messages if m["role"] == "user"]
    assert len(users) == 5
    assert users[0]["content"] == "turn 7"


def test_trim_keeps_the_system_prompt_pinned(executor: ToolExecutor) -> None:
    conversation = Conversation("s1")
    agent = Agent(FakeLLMClient([Reply(text="ok")] * 40), executor, max_turns=3)
    for i in range(10):
        agent.handle(f"turn {i}", conversation)
    assert conversation.messages[0]["role"] == "system"


# --- the audit record survives an outage ------------------------------------


def test_an_outage_after_a_tool_ran_still_reports_it(bank: SqliteBank) -> None:
    """`turns` must be right precisely when something went wrong.

    A transfer that executed before the endpoint died has to appear in the
    audit log; losing it makes the log wrong at the only moment it is read.
    """
    executor = ToolExecutor(bank)
    token = executor.execute(ToolCall("c0", "transfer_money", TRANSFER)).result["token"]

    class DiesAfterOneRound(LLMClient):
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages: Any, tools: Any) -> Reply:
            self.calls += 1
            if self.calls == 1:
                return Reply(
                    tool_calls=(
                        ToolCall("c1", "transfer_money", {**TRANSFER, "confirmation_token": token}),
                    )
                )
            raise LLMUnavailable("endpoint died")

    out = Agent(DiesAfterOneRound(), executor).handle("confirm it", Conversation("s1"))

    assert bank.balance("checking") == 243_350  # the money moved
    assert out.text == UNAVAILABLE
    assert [call.name for call in out.tool_calls] == ["transfer_money"]
    assert out.tool_results and out.tool_results[0]["status"] == "done"


# --- repeating the customer's own figure is not inventing one ---------------


def test_the_bot_may_repeat_an_amount_the_customer_just_said(
    executor: ToolExecutor,
) -> None:
    """Found in use: "transfer $400 to Ali" was answered with the fallback.

    The model asked for confirmation in words without calling the tool first.
    $400 came from the customer, so no tool had reported it, and the guard
    suppressed the reply -- leaving the conversation unable to proceed, since
    every "yes" produced the same suppression.
    """
    client = FakeLLMClient([Reply(text="I can transfer $400.00 to Ali. Shall I go ahead?")])
    out = Agent(client, executor).handle("Can you please transfer $400 to Ali?", Conversation("s1"))

    assert out.guard_hit is False
    assert "400" in out.text


def test_a_figure_the_customer_said_earlier_still_counts(executor: ToolExecutor) -> None:
    # The customer names the amount, the bot asks, the customer says "yes".
    conversation = Conversation("s1")
    client = FakeLLMClient(
        [
            Reply(text="I can transfer $400.00 to Ali. Shall I go ahead?"),
            Reply(text="Just to confirm, that's $400.00 to Ali?"),
        ]
    )
    agent = Agent(client, executor)
    agent.handle("transfer $400 to Ali", conversation)
    out = agent.handle("yes please", conversation)

    assert out.guard_hit is False


def test_a_pending_action_may_be_restated(bank: SqliteBank) -> None:
    executor = ToolExecutor(bank)
    executor.execute(ToolCall("c0", "transfer_money", TRANSFER))
    client = FakeLLMClient([Reply(text="I still have $50.00 to Sarah waiting. Go ahead?")])

    out = Agent(client, executor).handle("what was that again?", Conversation("s1"))

    assert out.guard_hit is False


def test_a_figure_from_nowhere_is_still_caught(executor: ToolExecutor) -> None:
    # The protection that matters must survive the fix.
    client = FakeLLMClient([Reply(text="Your savings balance is $31,415.00.")])
    out = Agent(client, executor).handle("what's my balance?", Conversation("s1"))

    assert out.guard_hit is True
    assert "31,415" not in out.text


# --- figures the bank gave earlier in the same conversation -----------------


def test_a_figure_from_an_earlier_lookup_can_be_restated(executor: ToolExecutor) -> None:
    """Found in use: asking about a balance twice failed the second time.

    The model answered the repeat from memory without calling the tool again.
    The figures were right and had come from the bank -- on the previous turn.
    """
    conversation = Conversation("s1")
    client = FakeLLMClient(
        [
            Reply(tool_calls=(ToolCall("c1", "list_accounts", {}),)),
            Reply(text="Checking is $2,483.50 and savings is $19,020.00."),
            Reply(text="Checking is $2,483.50 and savings is $19,020.00."),
        ]
    )
    agent = Agent(client, executor)

    first = agent.handle("what are my balances?", conversation)
    second = agent.handle("tell me again", conversation)

    assert first.guard_hit is False
    assert second.guard_hit is False, "a figure the bank already gave was suppressed"
    assert "2,483.50" in second.text


def test_memory_does_not_admit_a_figure_the_bank_never_gave(
    executor: ToolExecutor,
) -> None:
    conversation = Conversation("s1")
    client = FakeLLMClient(
        [
            Reply(tool_calls=(ToolCall("c1", "list_accounts", {}),)),
            Reply(text="Checking is $2,483.50."),
            Reply(text="Your savings balance is $31,415.00."),
        ]
    )
    agent = Agent(client, executor)
    agent.handle("balances?", conversation)
    out = agent.handle("and savings?", conversation)

    assert out.guard_hit is True


def test_a_suppressed_figure_does_not_become_quotable(executor: ToolExecutor) -> None:
    # The model's own text is never a source, or an invented figure would
    # launder itself simply by being repeated.
    conversation = Conversation("s1")
    client = FakeLLMClient(
        [
            Reply(text="Your balance is $31,415.00."),
            Reply(text="As I said, $31,415.00."),
        ]
    )
    agent = Agent(client, executor)
    agent.handle("balance?", conversation)
    out = agent.handle("say that again", conversation)

    assert out.guard_hit is True


def test_moving_money_forgets_the_balances_it_made_stale(bank: SqliteBank) -> None:
    """A balance reported before a transfer is no longer true after it.

    Remembering it would let the bot state yesterday's figure with full
    confidence, which is the failure this guard exists to prevent -- arriving
    by a different road.
    """
    executor = ToolExecutor(bank)
    conversation = Conversation("s1")
    client = FakeLLMClient(
        [
            Reply(tool_calls=(ToolCall("c1", "get_balance", {"account": "checking"}),)),
            Reply(text="Your checking balance is $2,483.50."),
            Reply(tool_calls=(ToolCall("c2", "transfer_money", TRANSFER),)),
            Reply(text="Transfer $50.00 to sarah. Shall I go ahead?"),
            Reply(text="Your checking balance is $2,483.50."),
        ]
    )
    agent = Agent(client, executor)
    agent.handle("balance?", conversation)
    agent.handle("send sarah fifty", conversation)
    executor.confirm()

    out = agent.handle("what's my balance now?", conversation)
    assert out.guard_hit is True, "the pre-transfer balance was still quotable"
