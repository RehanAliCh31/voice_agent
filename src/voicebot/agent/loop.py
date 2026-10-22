"""One conversational turn: ask the model, run what it asks for, check the answer.

The loop is bounded. A model that keeps calling tools is a model that is
stuck, and an unbounded loop against a shared endpoint is an outage rather
than a bug.

Nothing in here raises at its caller. A turn always produces something sayable,
because the alternative is a customer listening to silence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from voicebot.agent.executor import ToolExecutor
from voicebot.agent.guard import guard, money_in_results
from voicebot.llm.base import LLMClient, LLMUnavailable, Reply, ToolCall
from voicebot.llm.prompts import NOTHING_SAID, SYSTEM_PROMPT, UNAVAILABLE, UNVERIFIED
from voicebot.llm.tools import TOOL_SCHEMAS
from voicebot.utils.logging import get_logger

log = get_logger("agent.loop")

DEFAULT_MAX_TOOL_ROUNDS = 4
DEFAULT_MAX_TURNS = 20


@dataclass
class Conversation:
    """One caller's message history, in the shape the API expects."""

    session_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    turn_index: int = 0

    def append_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def append_assistant(self, reply: Reply) -> None:
        message: dict[str, Any] = {"role": "assistant", "content": reply.text}
        if reply.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in reply.tool_calls
            ]
        self.messages.append(message)

    def append_tool_result(self, call_id: str, result: dict[str, Any]) -> None:
        self.messages.append(
            {"role": "tool", "tool_call_id": call_id, "content": json.dumps(result)}
        )

    def trim(self, max_turns: int = DEFAULT_MAX_TURNS) -> None:
        """Bound the history, keeping the system prompt pinned.

        The context window is not the binding limit -- 262k tokens is far more
        than a conversation needs. Latency is: every message is re-sent on
        every call, so an unbounded history makes each turn slower than the
        last.
        """
        system = [m for m in self.messages[:1] if m.get("role") == "system"]
        rest = self.messages[len(system) :]
        if len(rest) > max_turns:
            rest = rest[-max_turns:]
        self.messages = system + rest


@dataclass(frozen=True)
class AgentResult:
    """Everything one turn produced."""

    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[dict[str, Any], ...] = ()
    guard_hit: bool = False
    suppressed: str | None = None
    rounds: int = 0


class Agent:
    """Drives one conversation against the model and its tools."""

    def __init__(
        self,
        client: LLMClient,
        executor: ToolExecutor,
        max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
        max_turns: int = DEFAULT_MAX_TURNS,
    ) -> None:
        self.client = client
        self.executor = executor
        self.max_tool_rounds = max_tool_rounds
        self.max_turns = max_turns

    # --- public --------------------------------------------------------------

    def handle(self, text: str, conversation: Conversation) -> AgentResult:
        """Advance the conversation by one turn."""
        self._ensure_system(conversation)
        conversation.turn_index += 1
        conversation.append_user(text)
        conversation.trim(self.max_turns)
        return self._run(conversation)

    def resolve_pending(self, confirmed: bool, conversation: Conversation) -> AgentResult:
        """Act on the Yes or Cancel button, then let the model say what happened.

        The result is appended to the history deliberately. Skipping it leaves
        a model whose context still says it is waiting for an answer, so the
        next "did that go through?" is answered from a history that is wrong.
        """
        self._ensure_system(conversation)
        pending = self.executor.pending
        result = self.executor.confirm() if confirmed else self.executor.cancel()

        call_id = "confirm"
        conversation.append_assistant(
            Reply(
                tool_calls=(
                    ToolCall(
                        id=call_id,
                        name=pending.tool if pending else "confirm",
                        arguments=pending.arguments if pending else {},
                    ),
                )
            )
        )
        conversation.append_tool_result(call_id, result)
        conversation.trim(self.max_turns)
        return self._run(conversation, seeded_results=[result])

    # --- internals -----------------------------------------------------------

    def _ensure_system(self, conversation: Conversation) -> None:
        if not conversation.messages or conversation.messages[0].get("role") != "system":
            conversation.messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    def _run(
        self, conversation: Conversation, seeded_results: list[dict[str, Any]] | None = None
    ) -> AgentResult:
        """Ask, execute, repeat -- until the model has words and wants nothing more."""
        calls: list[ToolCall] = []
        results: list[dict[str, Any]] = list(seeded_results or [])
        rounds = 0
        reply = Reply()

        for _ in range(self.max_tool_rounds):
            rounds += 1
            try:
                reply = self.client.complete(list(conversation.messages), list(TOOL_SCHEMAS))
            except LLMUnavailable as exc:
                log.warning("llm unavailable on round %d: %s", rounds, exc)
                return AgentResult(text=UNAVAILABLE, rounds=rounds)

            conversation.append_assistant(reply)

            if not reply.tool_calls:
                break

            for call in reply.tool_calls:
                result = self.executor.execute(call)
                calls.append(call)
                results.append(result)
                conversation.append_tool_result(call.id, result)

        conversation.trim(self.max_turns)
        text, hit, suppressed = self._finish(reply.text, results)
        return AgentResult(
            text=text,
            tool_calls=tuple(calls),
            tool_results=tuple(results),
            guard_hit=hit,
            suppressed=suppressed,
            rounds=rounds,
        )

    def _finish(
        self, text: str | None, results: list[dict[str, Any]]
    ) -> tuple[str, bool, str | None]:
        """Turn whatever the model left us with into something sayable, and check it."""
        spoken = (text or "").strip()
        if not spoken:
            # No words and no more rounds: say something rather than nothing.
            return NOTHING_SAID, False, None

        checked, hit = guard(spoken, results, self._fallback(results))
        return checked, hit, spoken if hit else None

    def _fallback(self, results: list[dict[str, Any]]) -> str:
        """What to say instead when the guard fires.

        Built from the tool results, so the customer still gets the real
        figure where there is one. Assembled from values, never generated --
        this is the one sentence that must be trustworthy by construction.
        """
        amounts = sorted(money_in_results(results))
        if not amounts:
            return UNVERIFIED
        if len(amounts) == 1:
            return f"The figure I have is ${amounts[0]:,.2f}."
        readable = ", ".join(f"${amount:,.2f}" for amount in amounts)
        return f"The figures I have are {readable}."
