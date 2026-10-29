"""The agent loop: the LLM, its tools, and the guard, in one turn.

Every test here runs against `FakeLLMClient`, so the whole loop — tool
calling, the confirmation gate, the guard, the failure paths — is exercised
with no endpoint and no network.
"""

from __future__ import annotations

from typing import Any

import pytest

from voicebot.agent.executor import ToolExecutor
from voicebot.agent.loop import Agent, Conversation, announces_action, asked_for_a_person
from voicebot.assistant.language import spoken_language
from voicebot.bank.sqlite_bank import SqliteBank
from voicebot.llm.base import FakeLLMClient, LLMClient, LLMUnavailable, Reply, ToolCall
from voicebot.llm.prompts import (
    HANDED_OVER,
    LANGUAGE_NOTE,
    PIN_NEEDED,
    SYSTEM_PROMPT,
    UNAVAILABLE,
    UNVERIFIED,
    in_language,
)
from voicebot.llm.tools import TOOL_SCHEMAS
from voicebot.messages import say

TRANSFER = {"amount_cents": 5_000, "payee": "sarah", "from_account": "checking"}
TO_ALI = {"amount_cents": 40_000, "payee": "ali", "from_account": "checking"}
MOVE = {"amount_cents": 25_000, "from_account": "checking", "to_account": "savings"}


@pytest.fixture
def executor(bank: SqliteBank) -> ToolExecutor:
    return ToolExecutor(bank)


def agent_with(replies: list[Reply], executor: ToolExecutor, **kwargs: Any) -> Agent:
    return Agent(FakeLLMClient(replies), executor, **kwargs)


def asked_for(text: str) -> Conversation:
    """A conversation in which the customer has said something.

    Settling a pending action against an empty one is a state production
    cannot reach -- whatever armed it, a voice turn or a card button, put the
    request in the history first. A real server answers a history with no
    user turn in it with 400, and the fake now does the same.
    """
    conversation = Conversation("s1")
    conversation.turn_index += 1
    conversation.append_user(text)
    return conversation


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
    assert len(client.calls[0][1]) == len(TOOL_SCHEMAS)


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
        def complete(self, messages: Any, tools: Any, tool_choice: str = "auto") -> Reply:
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
    # list_cards on purpose: it is the one lookup that reports no figures, so
    # the turn still ends with nothing the bot is entitled to say.
    out = agent_with(
        [
            Reply(text="Your balance is $500.00."),
            Reply(tool_calls=(ToolCall("c1", "list_cards", {}),)),
            Reply(text="Your balance is $500.00."),
        ],
        executor,
    ).handle("x", Conversation("s1"))

    assert out.guard_hit is True
    assert "500" not in out.text
    assert out.text == UNVERIFIED


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
    conversation = asked_for("send sarah fifty")
    client = FakeLLMClient([Reply(text="Done. $50.00 sent to sarah.")])

    # The money gate asks for a PIN after the yes. These tests are about
    # what happens once it is given, so it is given here the way the API
    # gives it: checked against the bank first, then said to have passed.
    executor.pin_accepted()
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

    Agent(client, executor).resolve_pending(False, asked_for("send sarah fifty"))

    assert bank.balance("checking") == 248_350
    assert executor.pending is None


def test_resolve_pending_with_nothing_pending_still_answers(executor: ToolExecutor) -> None:
    client = FakeLLMClient([Reply(text="There wasn't anything waiting.")])
    out = Agent(client, executor).resolve_pending(True, asked_for("yes"))
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
    # The money gate asks for a PIN after the yes. These tests are about
    # what happens once it is given, so it is given here the way the API
    # gives it: checked against the bank first, then said to have passed.
    executor.pin_accepted()

    class DiesAfterOneRound(LLMClient):
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages: Any, tools: Any, tool_choice: str = "auto") -> Reply:
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
    # Told it went through -- not "try again", which invites a second transfer.
    assert out.text == say("action_done", "en") and out.fallback == "action_done"
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

    What is under test is the guard: a figure the customer said themselves is
    theirs to hear back. The loop no longer ends a turn on a words-only
    question, so the round it forces is scripted rather than asserted here.
    """
    client = FakeLLMClient(
        [
            Reply(text="I can transfer $400.00 to Ali. Shall I go ahead?"),
            Reply(tool_calls=(ToolCall("c1", "transfer_money", TO_ALI),)),
            Reply(text="Transfer $400.00 to Ali from checking -- are you sure?"),
        ]
    )
    out = Agent(client, executor).handle("Can you please transfer $400 to Ali?", Conversation("s1"))

    assert out.guard_hit is False
    assert "400" in out.text


def test_a_figure_the_customer_said_earlier_still_counts(executor: ToolExecutor) -> None:
    # The customer names the amount, the bot asks, the customer says "yes".
    conversation = asked_for("something that armed it")
    client = FakeLLMClient(
        [
            Reply(tool_calls=(ToolCall("c1", "transfer_money", TO_ALI),)),
            Reply(text="Transfer $400.00 to Ali from checking -- are you sure?"),
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
    client = FakeLLMClient(
        [
            Reply(text="Your savings balance is $31,415.00."),
            Reply(tool_calls=(ToolCall("c1", "get_balance", {"account": "savings"}),)),
            Reply(text="Your savings balance is $31,415.00."),
        ]
    )
    out = Agent(client, executor).handle("what's my balance?", Conversation("s1"))

    assert out.guard_hit is True
    assert "31,415" not in out.text


# --- figures the bank gave earlier in the same conversation -----------------


def test_a_figure_from_an_earlier_lookup_can_be_restated(executor: ToolExecutor) -> None:
    """Found in use: asking about a balance twice failed the second time.

    The model answered the repeat from memory without calling the tool again.
    The figures were right and had come from the bank -- on the previous turn.
    """
    conversation = asked_for("something that armed it")
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
    conversation = asked_for("something that armed it")
    client = FakeLLMClient(
        [
            Reply(tool_calls=(ToolCall("c1", "list_accounts", {}),)),
            Reply(text="Checking is $2,483.50."),
            Reply(text="Your savings balance is $31,415.00."),
            Reply(tool_calls=(ToolCall("c2", "list_accounts", {}),)),
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
    conversation = asked_for("something that armed it")
    client = FakeLLMClient(
        [
            Reply(text="Your balance is $31,415.00."),
            Reply(tool_calls=(ToolCall("c1", "list_cards", {}),)),
            Reply(text="Your balance is $31,415.00."),
            Reply(text="As I said, $31,415.00."),
            Reply(tool_calls=(ToolCall("c2", "list_cards", {}),)),
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
    conversation = asked_for("something that armed it")
    client = FakeLLMClient(
        [
            Reply(tool_calls=(ToolCall("c1", "get_balance", {"account": "checking"}),)),
            Reply(text="Your checking balance is $2,483.50."),
            Reply(tool_calls=(ToolCall("c2", "transfer_money", TRANSFER),)),
            Reply(text="Transfer $50.00 to sarah. Shall I go ahead?"),
            Reply(text="Your checking balance is $2,483.50."),
            Reply(tool_calls=(ToolCall("c3", "list_cards", {}),)),
            Reply(text="Your checking balance is $2,483.50."),
        ]
    )
    agent = Agent(client, executor)
    agent.handle("balance?", conversation)
    agent.handle("send sarah fifty", conversation)
    # The money gate asks for a PIN after the yes; this test is about what
    # the transfer does to the balances remembered before it.
    executor.pin_accepted()
    executor.confirm()

    out = agent.handle("what's my balance now?", conversation)
    assert out.guard_hit is True, "the pre-transfer balance was still quotable"


def test_amounts_reach_the_page_as_figures(executor: ToolExecutor) -> None:
    # The model writes amounts as words because its replies are spoken; the
    # page shows figures, which are easier to check at a glance.
    spoken = "Checking holds two thousand four hundred eighty three dollars and fifty cents."
    client = FakeLLMClient(
        [
            Reply(tool_calls=(ToolCall("c1", "get_balance", {"account": "checking"}),)),
            Reply(text=spoken),
        ]
    )
    out = Agent(client, executor).handle("balance?", Conversation("s1"))

    assert out.guard_hit is False
    assert "$2,483.50" in out.text
    assert "thousand" not in out.text


# --- a change makes every earlier lookup out of date ------------------------


def test_paying_a_bill_invalidates_the_earlier_list(bank: SqliteBank) -> None:
    """Reported in use: a bill paid, then still shown as unpaid.

    The database was right; the model was reading the `list_bills` result
    still sitting in its context from before the payment. Asking it to look
    again is advice it follows most of the time, which is the wrong kind of
    guarantee for a bank. The stale result is now removed from what it sees.
    """
    executor = ToolExecutor(bank)
    conversation = asked_for("something that armed it")
    client = FakeLLMClient(
        [
            Reply(tool_calls=(ToolCall("c1", "list_bills", {}),)),
            Reply(text="You have five unpaid bills."),
            Reply(tool_calls=(ToolCall("c2", "pay_bill", {"bill": "electricity"}),)),
            Reply(text="Pay your electricity bill — are you sure?"),
            Reply(text="Done, your electricity bill is paid."),
        ]
    )
    agent = Agent(client, executor)
    agent.handle("what bills do I have?", conversation)
    agent.handle("pay electricity", conversation)
    # The money gate asks for a PIN after the yes. These tests are about
    # what happens once it is given, so it is given here the way the API
    # gives it: checked against the bank first, then said to have passed.
    executor.pin_accepted()
    agent.resolve_pending(True, conversation)

    tool_messages = [m for m in conversation.messages if m["role"] == "tool"]
    stale = [m for m in tool_messages if "superseded" in m["content"]]
    assert stale, "the pre-payment bill list was still readable"
    assert "electricity" not in stale[0]["content"], "the stale values survived"


def test_the_result_of_the_change_itself_is_not_invalidated(bank: SqliteBank) -> None:
    executor = ToolExecutor(bank)
    conversation = asked_for("something that armed it")
    client = FakeLLMClient([Reply(text="Done.")])
    executor.execute(ToolCall("c1", "pay_bill", {"bill": "electricity"}))

    # The money gate asks for a PIN after the yes. These tests are about
    # what happens once it is given, so it is given here the way the API
    # gives it: checked against the bank first, then said to have passed.
    executor.pin_accepted()
    Agent(client, executor).resolve_pending(True, conversation)

    last_tool = [m for m in conversation.messages if m["role"] == "tool"][-1]
    assert "superseded" not in last_tool["content"]
    assert "done" in last_tool["content"]


def test_a_read_only_lookup_does_not_invalidate_anything(bank: SqliteBank) -> None:
    executor = ToolExecutor(bank)
    conversation = asked_for("something that armed it")
    client = FakeLLMClient(
        [
            Reply(tool_calls=(ToolCall("c1", "list_bills", {}),)),
            Reply(text="Five unpaid."),
            Reply(tool_calls=(ToolCall("c2", "list_accounts", {}),)),
            Reply(text="Two accounts."),
        ]
    )
    agent = Agent(client, executor)
    agent.handle("bills?", conversation)
    agent.handle("accounts?", conversation)

    assert not [
        m for m in conversation.messages if m["role"] == "tool" and "superseded" in m["content"]
    ]


def test_a_suppressed_read_back_cancels_the_action_it_was_asking_about(
    bank: SqliteBank,
) -> None:
    """Reported in use, and the most serious thing found so far.

    The model proposed paying a bill, the guard suppressed its read-back, and
    the customer heard "let me check that properly" instead of the summary.
    The pending action survived, so their next "yes" -- meant for something
    else entirely -- paid $215.

    A read-back the customer never heard is not a read-back. If the sentence
    does not reach them, the action it was asking about must not survive.
    """
    executor = ToolExecutor(bank)
    client = FakeLLMClient(
        [
            Reply(tool_calls=(ToolCall("c1", "pay_bill", {"bill": "credit card"}),)),
            # A figure no tool returned: the guard will suppress this.
            Reply(text="Pay your credit card bill, $999.00 — are you sure?"),
        ]
    )
    out = Agent(client, executor).handle("what about the car bill?", Conversation("s1"))

    assert out.guard_hit is True
    assert executor.pending is None, "a read-back nobody heard left an action armed"
    assert bank.bills()[0].paid is False


def test_a_yes_after_a_suppressed_read_back_does_nothing(bank: SqliteBank) -> None:
    executor = ToolExecutor(bank)
    conversation = asked_for("something that armed it")
    client = FakeLLMClient(
        [
            Reply(tool_calls=(ToolCall("c1", "pay_bill", {"bill": "credit card"}),)),
            Reply(text="Pay your credit card bill, $999.00 — are you sure?"),
            Reply(text="There was nothing waiting."),
        ]
    )
    agent = Agent(client, executor)
    agent.handle("what about the car bill?", conversation)

    agent.resolve_pending(True, conversation)

    assert next(b for b in bank.bills() if b.name == "credit card").paid is False
    assert bank.balance("checking") == 248_350


def test_a_read_back_that_survives_the_guard_stays_armed(bank: SqliteBank) -> None:
    # The ordinary path must be untouched.
    executor = ToolExecutor(bank)
    client = FakeLLMClient(
        [
            Reply(tool_calls=(ToolCall("c1", "pay_bill", {"bill": "electricity"}),)),
            Reply(text="Pay your electricity bill, $72.50 — are you sure?"),
        ]
    )
    out = Agent(client, executor).handle("pay electricity", Conversation("s1"))

    assert out.guard_hit is False
    assert executor.pending is not None


# --- a promise is not an answer ----------------------------------------------
#
# Observed against the live model: asked about bills after a greeting, it
# replies "Let me pull up your bills." and calls nothing. The loop used to
# treat any reply without tool calls as a finished answer, so the customer
# heard the promise and the turn ended. Worse, when the conversation carried
# that promise forward the model went on to invent the bills outright.
#
# A turn may not end on an announcement. Detecting one is a guess; acting on
# it is not -- the retry forces the call rather than asking for it.


def test_a_promise_with_no_tool_call_forces_the_call(executor: ToolExecutor) -> None:
    agent = agent_with(
        [
            Reply(text="I can help with that. Let me pull up your bills."),
            Reply(tool_calls=(ToolCall("c1", "list_bills", {}),)),
            Reply(text="You have five unpaid bills."),
        ],
        executor,
    )

    out = agent.handle("tell me about my bills", Conversation("s1"))

    assert out.text == "You have five unpaid bills."
    assert [call.name for call in out.tool_calls] == ["list_bills"]


def test_the_forced_round_requires_a_tool(executor: ToolExecutor) -> None:
    """The retry must compel the call. Asking again politely is what failed."""
    client = FakeLLMClient(
        [
            Reply(text="Let me check that for you."),
            Reply(tool_calls=(ToolCall("c1", "list_bills", {}),)),
            Reply(text="You have five unpaid bills."),
        ]
    )
    Agent(client, executor).handle("my bills?", Conversation("s1"))

    assert client.choices == ["auto", "required", "auto"]


def test_a_question_back_is_left_alone(executor: ToolExecutor) -> None:
    """Asking which account is the behaviour we want, not a broken turn."""
    agent = agent_with([Reply(text="Which account would you like, checking or savings?")], executor)

    out = agent.handle("what is my balance?", Conversation("s1"))

    assert out.text == "Which account would you like, checking or savings?"
    assert out.rounds == 1


def test_a_greeting_is_left_alone(executor: ToolExecutor) -> None:
    agent = agent_with([Reply(text="Yes, I hear you clearly.")], executor)

    out = agent.handle("can you hear me?", Conversation("s1"))

    assert out.text == "Yes, I hear you clearly."
    assert out.rounds == 1


def test_a_promise_after_a_tool_already_ran_is_final(executor: ToolExecutor) -> None:
    """The work happened. Whatever it says now is a report, not a promise."""
    agent = agent_with(
        [
            Reply(tool_calls=(ToolCall("c1", "list_bills", {}),)),
            Reply(text="Let me know if you want me to pay any of them."),
        ],
        executor,
    )

    out = agent.handle("my bills?", Conversation("s1"))

    assert out.text == "Let me know if you want me to pay any of them."
    assert out.rounds == 2


def test_a_model_that_still_will_not_call_is_not_left_mid_promise(
    executor: ToolExecutor,
) -> None:
    """The forced round can come back empty. The customer must not hear the promise."""
    agent = agent_with(
        [
            Reply(text="Let me pull up your bills."),
            Reply(text="Let me pull up your bills."),
        ],
        executor,
    )

    out = agent.handle("my bills?", Conversation("s1"))

    assert "Let me pull up" not in out.text


# --- asking for a yes with nothing to act on ---------------------------------
#
# Found against the live model on a deposit: it said "Deposit $250.00 into
# checking by cheque -- are you sure?" and called nothing. The sentence is a
# correct read-back in every respect except that no action was armed behind
# it, so the customer's yes had nothing to agree to and the money never moved.
#
# A promise ending in a question mark is still a promise. What makes this one
# safe to act on is not the wording: it is that the executor knows whether
# anything is pending, and nothing was.


def test_a_read_back_with_nothing_armed_forces_the_call(executor: ToolExecutor) -> None:
    agent = agent_with(
        [
            Reply(text="Transfer $50.00 to Sarah from checking -- are you sure?"),
            Reply(tool_calls=(ToolCall("c1", "transfer_money", TRANSFER),)),
            Reply(text="Transfer $50.00 to Sarah from checking -- are you sure?"),
        ],
        executor,
    )

    out = agent.handle("send sarah fifty", Conversation("s1"))

    assert [call.name for call in out.tool_calls] == ["transfer_money"]
    assert executor.pending is not None, "the yes now has something to act on"


def test_a_real_read_back_is_left_alone(executor: ToolExecutor) -> None:
    """Something is armed, so the question is the gate working, not a promise."""
    agent = agent_with(
        [
            Reply(tool_calls=(ToolCall("c1", "transfer_money", TRANSFER),)),
            Reply(text="Transfer $50.00 to Sarah from checking -- are you sure?"),
        ],
        executor,
    )

    out = agent.handle("send sarah fifty", Conversation("s1"))

    assert out.rounds == 2
    assert out.text.endswith("are you sure?")


def test_a_repeated_read_back_while_armed_is_left_alone(executor: ToolExecutor) -> None:
    """No tool ran this turn, but one is still pending from the last."""
    agent = agent_with(
        [
            Reply(tool_calls=(ToolCall("c1", "transfer_money", TRANSFER),)),
            Reply(text="Transfer $50.00 to Sarah from checking -- are you sure?"),
            Reply(text="Just to check -- transfer $50.00 to Sarah, are you sure?"),
        ],
        executor,
    )
    conversation = Conversation("s1")
    agent.handle("send sarah fifty", conversation)

    out = agent.handle("sorry, what?", conversation)

    assert out.rounds == 1
    assert "are you sure" in out.text


def test_a_clarifying_question_is_still_left_alone(executor: ToolExecutor) -> None:
    """No amount in it, so there is nothing it could be asking agreement for."""
    agent = agent_with(
        [Reply(text="Which account would you like me to check, checking or savings?")],
        executor,
    )

    out = agent.handle("what is my balance?", Conversation("s1"))

    assert out.rounds == 1
    assert out.tool_calls == ()


def test_an_offer_after_a_lookup_is_left_alone(executor: ToolExecutor) -> None:
    """The dangerous false positive: forcing a call here could pay a bill."""
    agent = agent_with(
        [
            Reply(tool_calls=(ToolCall("c1", "list_bills", {}),)),
            Reply(text="Your water bill is $31.00. Would you like me to pay it?"),
        ],
        executor,
    )

    out = agent.handle("what do I owe?", Conversation("s1"))

    assert out.rounds == 2
    assert [call.name for call in out.tool_calls] == ["list_bills"]
    assert executor.pending is None, "nothing may be armed by an offer alone"


def test_the_unfinished_reply_is_struck_from_the_history(executor: ToolExecutor) -> None:
    """Observed live: every forced deposit was read back as "Yes".

    The model's own unkept question stayed in the history, so on the next
    round it found "are you sure?" above a tool result and answered itself
    rather than the customer. The customer never heard that sentence -- the
    turn returns only its final reply -- so it does not belong in the record.
    """
    conversation = Conversation("s1")
    agent = agent_with(
        [
            Reply(text="Move $250.00 from checking to savings -- are you sure?"),
            Reply(tool_calls=(ToolCall("c1", "move_money", MOVE),)),
            Reply(text="Move $250.00 from checking to savings -- are you sure?"),
        ],
        executor,
    )

    agent.handle("move two fifty into savings", conversation)

    spoken = [m.get("content") for m in conversation.messages if m.get("role") == "assistant"]
    assert spoken.count("Move $250.00 from checking to savings -- are you sure?") == 1


def test_a_spoken_amount_reaches_the_page_as_a_figure(executor: ToolExecutor) -> None:
    """The ledger shows $7.60; the sentence beside it should not say seven sixty."""
    agent = agent_with(
        [
            Reply(
                tool_calls=(
                    ToolCall(
                        "c1", "list_transactions", {"period": "yesterday", "merchant": "starbucks"}
                    ),
                )
            ),
            Reply(text="Starbucks, seven sixty, yesterday."),
        ],
        executor,
    )

    out = agent.handle("what did I spend at starbucks?", Conversation("s1"))

    assert out.text == "Starbucks, $7.60, yesterday."
    assert out.guard_hit is False


# --- a figure stated with nothing looked up ----------------------------------
#
# Found live: asked what it had paid Netflix, the model answered "$15.49, on
# the 3rd" with no tool call at all, three times in four. The real charge is
# $10.99. The guard caught the invention every time -- the customer never
# heard $15.49 -- but what they heard instead was an apology, when the answer
# was one lookup away.
#
# The guard's job is to stop the wrong figure. Fetching the right one is this
# loop's job, and the signal is exact: money stated, and nothing looked up to
# state it from.


def test_an_invented_figure_sends_the_model_to_look_it_up(executor: ToolExecutor) -> None:
    agent = agent_with(
        [
            Reply(text="You've paid Netflix $15.49 this month."),
            Reply(
                tool_calls=(
                    ToolCall(
                        "c1", "list_transactions", {"period": "this week", "merchant": "netflix"}
                    ),
                )
            ),
            Reply(text="You've paid Netflix $10.99."),
        ],
        executor,
    )

    out = agent.handle("what have I paid netflix?", Conversation("s1"))

    assert out.text == "You've paid Netflix $10.99."
    assert out.guard_hit is False
    assert [call.name for call in out.tool_calls] == ["list_transactions"]


def test_a_figure_invented_twice_is_still_recorded_as_a_guard_hit(
    executor: ToolExecutor,
) -> None:
    """The metric is how anyone notices the model reaching for figures."""
    agent = agent_with(
        [
            Reply(text="You've paid Netflix $15.49 this month."),
            Reply(
                tool_calls=(
                    ToolCall(
                        "c1", "list_transactions", {"period": "this week", "merchant": "netflix"}
                    ),
                )
            ),
            Reply(text="You've paid Netflix $15.49 this month."),
        ],
        executor,
    )

    out = agent.handle("what have I paid netflix?", Conversation("s1"))

    assert out.guard_hit is True
    assert "15.49" not in out.text


def test_a_figure_the_customer_said_is_not_a_lookup_owed(executor: ToolExecutor) -> None:
    """Quoting them back is not inventing. It must not trigger a lookup."""
    agent = agent_with([Reply(text="That's $400.00 to Ali, from checking.")], executor)

    out = agent.handle("transfer $400 to Ali", Conversation("s1"))

    assert out.rounds == 1
    assert out.tool_calls == ()


def test_a_reply_with_no_figures_owes_nothing(executor: ToolExecutor) -> None:
    agent = agent_with([Reply(text="I'm well, thanks. How can I help?")], executor)

    out = agent.handle("how are you?", Conversation("s1"))

    assert out.rounds == 1


# --- asking for a person is not a request the model gets to decline ----------
#
# Every source agreed on two things: a handoff must be able to fire by rule
# rather than by the model's confidence, and a customer who wants a human and
# cannot get one loses more trust than the bot ever saves. So this is enforced
# here, the way the confirmation gate is -- the model is told to hand off, and
# the code does it whether or not the model agreed.


def test_asking_for_a_person_hands_off_even_if_the_model_ignores_it(
    bank: SqliteBank, executor: ToolExecutor
) -> None:
    agent = agent_with([Reply(text="I can help you with that myself!")], executor)

    out = agent.handle("I want to speak to a human", Conversation("s1"))

    assert bank.handoffs(), "the model declined; the rule did not"
    assert bank.handoffs()[0].reason == "customer asked"
    assert out.handed_off is not None


def test_the_reply_gives_the_customer_the_reference(
    bank: SqliteBank, executor: ToolExecutor
) -> None:
    agent = agent_with([Reply(text="I can help you with that myself!")], executor)

    out = agent.handle("put me through to a real person", Conversation("s1"))

    assert bank.handoffs()[0].reference in out.text


def test_an_ordinary_question_hands_off_nothing(bank: SqliteBank, executor: ToolExecutor) -> None:
    agent = agent_with([Reply(text="Your balance is fine.")], executor)

    agent.handle("what is my checking balance?", Conversation("s1"))

    assert bank.handoffs() == ()


def test_the_model_handing_off_itself_opens_exactly_one_case(
    bank: SqliteBank, executor: ToolExecutor
) -> None:
    """The rule must not raise a second case on top of the model's."""
    agent = agent_with(
        [
            Reply(tool_calls=(ToolCall("c1", "hand_off", {"reason": "fraud"}),)),
            Reply(text="I'm putting you through to our fraud team."),
        ],
        executor,
    )

    agent.handle("someone used my card, I want to speak to a person", Conversation("s1"))

    assert len(bank.handoffs()) == 1
    assert bank.handoffs()[0].reason == "fraud"


def test_a_handoff_by_rule_drops_an_armed_transfer(
    bank: SqliteBank, executor: ToolExecutor
) -> None:
    conversation = Conversation("s1")
    agent = agent_with(
        [
            Reply(tool_calls=(ToolCall("c1", "transfer_money", TRANSFER),)),
            Reply(text="Transfer $50.00 to Sarah from checking -- are you sure?"),
            Reply(text="Of course."),
        ],
        executor,
    )
    agent.handle("send sarah fifty", conversation)
    assert executor.pending is not None

    agent.handle("actually, let me talk to a person", conversation)

    assert executor.pending is None
    assert bank.balance("checking") == 248_350


# --- in Urdu -----------------------------------------------------------------
#
# The bot follows the customer into Urdu. The model is reminded of it on every
# Urdu turn, and what the bot says without the model -- an apology, a
# handoff, a figure it would not repeat -- is said in Urdu as well.


def in_urdu() -> Conversation:
    conversation = Conversation("s1")
    conversation.language = "ur"
    return conversation


def test_each_turn_tells_the_model_the_language_to_answer_in(executor: ToolExecutor) -> None:
    client = FakeLLMClient([Reply(text="میں ٹھیک ہوں۔"), Reply(text="I'm well.")])
    agent = Agent(client, executor)
    conversation = in_urdu()

    agent.handle("آپ کیسے ہیں", conversation)
    conversation.language = "en"
    agent.handle("how are you?", conversation)

    # Measured: without the note a spoken Urdu question was answered in Urdu
    # written in Latin letters, and English after two Urdu turns in Urdu.
    first, second = client.calls[0][0], client.calls[1][0]
    assert first[-1]["content"] == f"آپ کیسے ہیں\n\n({LANGUAGE_NOTE['ur']})"
    assert second[-1]["content"] == f"how are you?\n\n({LANGUAGE_NOTE['en']})"
    # Only the latest message carries one, and only in what is sent: the
    # history keeps the customer's words as they were said.
    assert second[1]["content"] == "آپ کیسے ہیں"
    said = [m["content"] for m in conversation.messages if m["role"] == "user"]
    assert said == ["آپ کیسے ہیں", "how are you?"]
    # The system prompt is the same on every turn.
    assert first[0] == second[0] == {"role": "system", "content": SYSTEM_PROMPT}


def test_the_note_stays_on_the_customers_message_through_a_tool_round(
    executor: ToolExecutor,
) -> None:
    client = FakeLLMClient(
        [
            Reply(tool_calls=(ToolCall("c1", "get_balance", {"account": "savings"}),)),
            Reply(text="آپ کے سیونگز اکاؤنٹ میں $19,020.00 ہیں۔"),
        ]
    )
    Agent(client, executor).handle("میرا سیونگز بیلنس", in_urdu())

    after_the_tool = client.calls[1][0]
    assert after_the_tool[-1]["role"] == "tool"
    assert after_the_tool[1]["content"].endswith(f"({LANGUAGE_NOTE['ur']})")


@pytest.mark.parametrize(
    "promise",
    [
        # Seen live, with nothing called after it.
        "میں آپ کے چیکنگ اکاؤنٹ میں موجود رقم دیکھتا ہوں۔ براہ کرم انتظار کریں۔",
        "میں آپ کے لیے چیک کروں گا۔",
        "میں ابھی چیک کر رہا ہوں۔",
        "ایک لمحہ، میں دیکھ لیتا ہوں۔",
        "میں آپ کا بیلنس دیکھ لوں گا۔",
    ],
)
def test_an_urdu_promise_is_a_promise(promise: str) -> None:
    assert announces_action(promise)


@pytest.mark.parametrize(
    "reply",
    [
        # A question, with the question mark Urdu is written with.
        "میں آپ کے لیے چیک کر سکتا ہوں، کون سا اکاؤنٹ؟",
        # The customer can look; the bot promised nothing.
        "آپ اپنی ایپ میں دیکھ سکتے ہیں۔",
        # Done, not promised.
        "میں نے دیکھا کہ آپ کے تمام بل ادا ہو چکے ہیں۔",
        "اسلام آباد پاکستان کا دارالحکومت ہے۔",
    ],
)
def test_urdu_that_promises_nothing_is_left_alone(reply: str) -> None:
    assert not announces_action(reply)


def test_an_urdu_promise_forces_the_call(executor: ToolExecutor) -> None:
    client = FakeLLMClient(
        [
            Reply(text="میں آپ کے بل دیکھتا ہوں۔ براہ کرم انتظار کریں۔"),
            Reply(tool_calls=(ToolCall("c1", "list_bills", {}),)),
            Reply(text="آپ کے نو بل باقی ہیں۔"),
        ]
    )

    out = Agent(client, executor).handle("میرے بل", in_urdu())

    assert client.choices == ["auto", "required", "auto"]
    assert out.text == "آپ کے نو بل باقی ہیں۔"


def test_an_outage_is_apologised_for_in_urdu(executor: ToolExecutor) -> None:
    class Down(LLMClient):
        def complete(self, messages: Any, tools: Any, tool_choice: str = "auto") -> Reply:
            raise LLMUnavailable("down")

    out = Agent(Down(), executor).handle("میرا بیلنس", in_urdu())

    assert out.text == in_language(UNAVAILABLE, "ur") != UNAVAILABLE


def test_a_figure_the_bank_never_gave_is_replaced_in_urdu(executor: ToolExecutor) -> None:
    client = FakeLLMClient(
        [
            Reply(tool_calls=(ToolCall("c1", "get_balance", {"account": "savings"}),)),
            Reply(text="آپ کے سیونگز اکاؤنٹ میں $31,415.00 ہیں۔"),
        ]
    )

    out = Agent(client, executor).handle("میرا سیونگز بیلنس", in_urdu())

    assert out.guard_hit is True
    assert "31,415" not in out.text
    assert "$19,020.00" in out.text and spoken_language(out.text) == "ur"


def test_asking_for_a_person_in_urdu_is_answered_in_urdu(
    bank: SqliteBank, executor: ToolExecutor
) -> None:
    agent = agent_with([Reply(text="میں خود آپ کی مدد کر سکتا ہوں!")], executor)

    out = agent.handle("مجھے کسی انسان سے بات کرنی ہے", in_urdu())

    reference = bank.handoffs()[0].reference
    assert out.text == in_language(HANDED_OVER, "ur").format(reference=reference)


@pytest.mark.parametrize(
    ("said", "wants"),
    [
        ("مجھے کسی انسان سے بات کرنی ہے", True),
        ("نمائندے سے بات کروائیں", True),
        ("کسٹمر سروس سے ملائیں", True),
        ("انسان سے غلطی ہو جاتی ہے", False),  # people make mistakes
        ("میرا بیلنس کیا ہے", False),
        # Typed in Latin letters, as Urdu often is.
        ("mujhe kisi insaan se baat karni hai", True),
        ("agent se baat karao", True),
        ("insaan se ghalti ho jati hai", False),
    ],
)
def test_asking_for_a_person_in_urdu(said: str, wants: bool) -> None:
    assert asked_for_a_person(said) is wants


def test_waiting_on_the_pin_is_said_in_urdu(executor: ToolExecutor) -> None:
    agent = agent_with(
        [
            Reply(tool_calls=(ToolCall("c1", "pay_bill", {"bill": "internet"}),)),
            Reply(text="انٹرنیٹ کا بل $40.00 ادا کر دوں، کیا آپ کو یقین ہے؟"),
        ],
        executor,
    )
    conversation = in_urdu()
    agent.handle("میرا انٹرنیٹ کا بل ادا کر دیں", conversation)

    out = agent.resolve_pending(True, conversation)

    assert out.text == in_language(PIN_NEEDED, "ur") != PIN_NEEDED
