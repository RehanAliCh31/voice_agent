"""Fault injection: the bank and the model failing inside a turn.

What every test here checks, in one way or another: the customer hears a
sentence that is true -- never a guess, never "try again" about money that
already moved -- nothing raises out of the turn, and the history is left
whole, so the next turn is not refused by the server for a dangling call.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from voicebot.agent.executor import OUTCOME_UNKNOWN, TOOL_FAILED, ToolExecutor
from voicebot.agent.loop import Agent, Conversation
from voicebot.bank.base import StorageError
from voicebot.bank.sqlite_bank import SqliteBank
from voicebot.errors import Cancelled
from voicebot.llm.base import (
    ContextOverflow,
    FakeLLMClient,
    LLMClient,
    LLMUnavailable,
    Reply,
    ToolCall,
)
from voicebot.messages import say
from voicebot.utils import observability
from voicebot.utils.resilience import CancelToken, cancel_context

TRANSFER = {"amount_cents": 5_000, "payee": "sarah", "from_account": "checking"}


class Faulty:
    """The bank, with the named methods failing the way a lost database does."""

    def __init__(self, bank: SqliteBank, *failing: str, error: Exception | None = None) -> None:
        self._bank = bank
        self._failing = set(failing)
        self._error = error

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._bank, name)
        if name in self._failing and callable(attr):

            def broken(*args: Any, **kwargs: Any) -> Any:
                raise self._error or StorageError(f"{name}: server closed the connection")

            return broken
        return attr


def _whole(conversation: Conversation) -> None:
    """Every tool call in the history has its result: the server would refuse it otherwise."""
    asked = {
        call["id"]
        for m in conversation.messages
        if m.get("role") == "assistant"
        for call in m.get("tool_calls") or ()
    }
    answered = {m.get("tool_call_id") for m in conversation.messages if m.get("role") == "tool"}
    assert asked <= answered, f"dangling tool calls: {asked - answered}"


def _in_urdu() -> Conversation:
    conversation = Conversation("s1")
    conversation.language = "ur"
    return conversation


# --- the bank failing under a lookup --------------------------------------------


@pytest.mark.parametrize("language", ["en", "ur"])
def test_a_lookup_the_bank_cannot_answer_is_said_and_never_guessed(
    bank: SqliteBank, language: str
) -> None:
    client = FakeLLMClient(
        [Reply(tool_calls=(ToolCall("c1", "get_balance", {"account": "savings"}),))]
    )
    conversation = Conversation("s1")
    conversation.language = language

    out = Agent(client, ToolExecutor(Faulty(bank, "balance", "accounts"))).handle(  # type: ignore[arg-type]
        "what is my savings balance", conversation
    )

    assert out.text == say("tool_failed", language)
    assert out.fallback == TOOL_FAILED
    # The model was not asked what to make of the failure: it would have guessed.
    assert len(client.calls) == 1
    assert "$" not in out.text
    _whole(conversation)
    assert observability.error_counts()[("bank", "database_error")] == 1


def test_one_failed_tool_of_two_leaves_no_call_without_a_result(bank: SqliteBank) -> None:
    client = FakeLLMClient(
        [
            Reply(
                tool_calls=(
                    ToolCall("c1", "list_bills", {}),
                    ToolCall("c2", "get_balance", {"account": "checking"}),
                )
            )
        ]
    )
    conversation = Conversation("s1")
    out = Agent(client, ToolExecutor(Faulty(bank, "bills"))).handle("bills?", conversation)  # type: ignore[arg-type]

    assert out.fallback == TOOL_FAILED
    _whole(conversation)


def test_a_tool_that_crashes_is_reported_and_the_turn_still_ends_in_a_sentence(
    bank: SqliteBank,
) -> None:
    client = FakeLLMClient([Reply(tool_calls=(ToolCall("c1", "list_cards", {}),))])
    executor = ToolExecutor(Faulty(bank, "cards", error=TypeError("unexpected None")))  # type: ignore[arg-type]

    out = Agent(client, executor).handle("my cards", Conversation("s1"))

    assert out.text == say("tool_failed", "en")
    assert observability.error_counts()[("list_cards", "tool_error")] == 1


# --- the bank failing under a write ----------------------------------------------


def test_a_write_that_fails_part_way_is_not_said_to_have_changed_nothing(
    bank: SqliteBank,
) -> None:
    """It may have landed. "Nothing has changed" could make the customer pay twice."""
    executor = ToolExecutor(Faulty(bank, "transfer"))  # type: ignore[arg-type]
    executor.execute(ToolCall("c0", "transfer_money", TRANSFER))
    executor.pin_accepted()
    conversation = Conversation("s1")
    conversation.append_user("send sarah fifty")
    writes = executor.writes
    client = FakeLLMClient([])

    out = Agent(client, executor).resolve_pending(True, conversation)

    assert out.text == say("outcome_unknown", "en")
    assert out.fallback == OUTCOME_UNKNOWN
    assert client.calls == [], "the model is not asked to phrase it"
    assert executor.writes == writes + 1, "every figure read before it is now stale"
    assert executor.pending is None, "and it is not left armed to be agreed to again"
    _whole(conversation)


def test_the_pin_card_unreadable_keeps_the_action_for_later(bank: SqliteBank) -> None:
    executor = ToolExecutor(bank)
    executor.execute(ToolCall("c0", "pay_bill", {"bill": "internet"}))
    executor.bank = Faulty(bank, "cards")  # type: ignore[assignment]

    out = executor.confirm()

    assert out.failed == TOOL_FAILED
    assert executor.pending is not None


# --- the model failing after the money moved ---------------------------------------


@pytest.mark.parametrize(("confirmed", "key"), [(True, "action_done"), (False, "action_cancelled")])
def test_the_model_down_after_yes_or_cancel_still_says_what_happened(
    bank: SqliteBank, confirmed: bool, key: str
) -> None:
    class Down(LLMClient):
        def complete(self, messages: Any, tools: Any, tool_choice: str = "auto") -> Reply:
            raise LLMUnavailable("down")

    executor = ToolExecutor(bank)
    executor.execute(ToolCall("c0", "pay_bill", {"bill": "internet"}))
    executor.pin_accepted()
    conversation = _in_urdu()
    conversation.append_user("میرا انٹرنیٹ کا بل ادا کر دیں")

    out = Agent(Down(), executor).resolve_pending(confirmed, conversation)

    assert out.text == say(key, "ur")
    _whole(conversation)


# --- time, length and leaving -----------------------------------------------------------


def test_a_turn_that_runs_out_of_budget_ends_rather_than_starting_another_round(
    bank: SqliteBank,
) -> None:
    class Slow(LLMClient):
        def __init__(self) -> None:
            self.rounds = 0

        def complete(self, messages: Any, tools: Any, tool_choice: str = "auto") -> Reply:
            self.rounds += 1
            time.sleep(0.4)
            return Reply(tool_calls=(ToolCall(f"c{self.rounds}", "list_accounts", {}),))

    client = Slow()
    started = time.monotonic()
    out = Agent(client, ToolExecutor(bank), max_tool_rounds=10, turn_budget_s=1.5).handle(
        "everything", Conversation("s1")
    )

    assert time.monotonic() - started < 2.0
    assert client.rounds < 10
    assert out.fallback == "unavailable"
    assert observability.error_counts()[("agent", "provider_timeout")] == 1


def test_a_conversation_too_long_for_the_model_is_trimmed_and_asked_again(
    bank: SqliteBank,
) -> None:
    class TooLongOnce(LLMClient):
        def __init__(self) -> None:
            self.sent: list[int] = []

        def complete(self, messages: Any, tools: Any, tool_choice: str = "auto") -> Reply:
            self.sent.append(len(messages))
            if len(self.sent) == 1:
                raise ContextOverflow("too long")
            return Reply(text="Hello again.")

    conversation = Conversation("s1")
    for n in range(8):
        conversation.append_user(f"turn {n}")
        conversation.append_assistant(Reply(text=f"reply {n}"))
    client = TooLongOnce()

    out = Agent(client, ToolExecutor(bank)).handle("hi", conversation)

    assert out.text == "Hello again."
    assert client.sent[1] < client.sent[0]


def test_rounds_running_out_on_the_call_that_armed_an_action_read_it_back(
    bank: SqliteBank,
) -> None:
    client = FakeLLMClient([Reply(tool_calls=(ToolCall("c1", "transfer_money", TRANSFER),))])
    executor = ToolExecutor(bank)

    out = Agent(client, executor, max_tool_rounds=1).handle("send sarah fifty", Conversation("s1"))

    assert executor.pending is not None
    assert out.text == f"{executor.pending.summary}?"
    assert out.fallback is None


def test_a_caller_who_left_stops_the_turn_between_rounds_with_the_history_whole(
    bank: SqliteBank,
) -> None:
    token = CancelToken()

    class LeavesAfterOne(LLMClient):
        def complete(self, messages: Any, tools: Any, tool_choice: str = "auto") -> Reply:
            token.cancel("the caller hung up")
            return Reply(tool_calls=(ToolCall("c1", "list_accounts", {}),))

    conversation = Conversation("s1")
    with cancel_context(token), pytest.raises(Cancelled, match="hung up before model round"):
        Agent(LeavesAfterOne(), ToolExecutor(bank)).handle("balances", conversation)
    _whole(conversation)


# --- asking for a person when the bank is down, and asking twice ----------------------------


def test_asking_for_a_person_with_the_bank_down_gives_another_way_through(
    bank: SqliteBank,
) -> None:
    executor = ToolExecutor(Faulty(bank, "open_handoff"))  # type: ignore[arg-type]
    agent = Agent(FakeLLMClient([Reply(text="I can help!")]), executor)

    out = agent.handle("I want to speak to a human", Conversation("s1"))

    assert out.text == say("cannot_complete", "en")


def test_a_handoff_asked_for_twice_is_one_case(bank: SqliteBank) -> None:
    executor = ToolExecutor(bank)
    first = executor.execute(ToolCall("c1", "hand_off", {"reason": "customer asked"}))
    second = executor.execute(ToolCall("c2", "hand_off", {"reason": "customer asked"}))

    assert first.result["reference"] == second.result["reference"]
    assert len(bank.handoffs()) == 1


# --- what the model sends that only looks right ----------------------------------------


def test_the_word_false_is_not_true(bank: SqliteBank) -> None:
    executor = ToolExecutor(bank)
    out = executor.execute(
        ToolCall(
            "c1",
            "freeze_card",
            {"last4": "4417", "report_fraud": "false", "order_replacement": "false"},
        )
    )
    assert executor.pending is not None
    assert executor.pending.arguments["report_fraud"] is False
    assert out.result["status"] == "needs_confirmation"


def test_a_flag_that_is_neither_is_refused_with_a_hint(bank: SqliteBank) -> None:
    out = ToolExecutor(bank).execute(
        ToolCall("c1", "freeze_card", {"last4": "4417", "report_fraud": "maybe"})
    )
    assert "true or false" in out.result["error"]


def test_in_urdu_the_read_back_points_to_the_screen_rather_than_speak_english(
    bank: SqliteBank,
) -> None:
    client = FakeLLMClient([Reply(tool_calls=(ToolCall("c1", "transfer_money", TRANSFER),))])
    executor = ToolExecutor(bank)
    conversation = Conversation("s1")
    conversation.language = "ur"

    out = Agent(client, executor, max_tool_rounds=1).handle("سارہ کو پچاس بھیج دیں", conversation)

    # Still armed for the Yes button; the English summary is on the screen.
    assert executor.pending is not None
    assert out.text == say("readback_on_screen", "ur")
