"""One conversational turn: ask the model, run what it asks for, check the answer.

The loop is bounded. A model that keeps calling tools is a model that is
stuck, and an unbounded loop against a shared endpoint is an outage rather
than a bug.

Nothing in here raises at its caller. A turn always produces something sayable,
because the alternative is a customer listening to silence.
"""

from __future__ import annotations

import json
import secrets
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from voicebot.agent.executor import ToolExecutor, ToolOutcome
from voicebot.agent.guard import fallback_sentence, guard, in_digits, money_in_text
from voicebot.llm.base import LLMClient, LLMUnavailable, Reply, ToolCall
from voicebot.llm.prompts import NOTHING_SAID, SYSTEM_PROMPT, UNAVAILABLE
from voicebot.llm.tools import TOOL_SCHEMAS
from voicebot.utils.logging import get_logger

log = get_logger("agent.loop")

DEFAULT_MAX_TOOL_ROUNDS = 4
DEFAULT_MAX_TURNS = 20

# How many turns a figure the bank gave stays quotable without a fresh lookup.
RECALL_TURNS = 6


@dataclass
class Conversation:
    """One caller's message history, in the shape the API expects."""

    session_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    turn_index: int = 0
    # Figures the bank has reported in this conversation, with the turn they
    # came from, so the bot can restate a balance it looked up a moment ago
    # without querying again. Only tool results land here -- never the model's
    # own words, or an invented figure could launder itself by repetition.
    sourced: list[tuple[int, int, int]] = field(default_factory=list)

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

    def remember(self, cents: Sequence[int], writes: int) -> None:
        """Record figures the bank gave on this turn, and what it knew then."""
        self.sourced.extend((self.turn_index, writes, value) for value in cents)

    def recall(self, writes: int, within_turns: int = RECALL_TURNS) -> tuple[int, ...]:
        """Figures still worth repeating: recent, and not made stale by a write.

        A balance reported before a transfer is not true after it. Quoting it
        would be the bot stating a confident, wrong figure -- the exact failure
        the guard exists to prevent, arriving by another road -- so anything
        recorded before the latest write is dropped.
        """
        cutoff = self.turn_index - within_turns
        return tuple(value for turn, at, value in self.sourced if turn >= cutoff and at == writes)

    def trim(self, max_turns: int = DEFAULT_MAX_TURNS) -> None:
        """Bound the history to whole turns, keeping the system prompt pinned.

        Cutting by message count would eventually leave a `tool` message whose
        assistant parent had been dropped. An OpenAI-compatible server rejects
        that with a 400, which this client treats as non-retryable, so the
        customer would hear the outage apology until the orphan scrolled out.
        A turn starts at a `user` message and runs to just before the next
        one, so slicing on those boundaries can never split a tool call from
        its result.

        The context window is not the binding limit -- 262k tokens is far more
        than a conversation needs. Latency is: every message is re-sent on
        every call, so an unbounded history makes each turn slower than the
        last.
        """
        system = [m for m in self.messages[:1] if m.get("role") == "system"]
        rest = self.messages[len(system) :]

        starts = [i for i, m in enumerate(rest) if m.get("role") == "user"]
        if len(starts) <= max_turns:
            return
        self.messages = system + rest[starts[-max_turns] :]


@dataclass(frozen=True)
class AgentResult:
    """Everything one turn produced."""

    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[dict[str, Any], ...] = ()
    sourced_cents: tuple[int, ...] = ()
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
        outcome = self.executor.confirm() if confirmed else self.executor.cancel()

        # A real tool name and a fresh id: duplicate ids across a session, or
        # a name absent from the schemas, are the kind of thing a stricter
        # server rejects -- on the one path that moves money.
        call_id = f"confirm-{secrets.token_urlsafe(6)}"
        injected = ToolCall(
            id=call_id,
            name=pending.tool if pending else "pay_bill",
            arguments=dict(pending.arguments) if pending else {},
        )
        conversation.append_assistant(Reply(tool_calls=(injected,)))
        conversation.remember(outcome.sourced_cents, self.executor.writes)
        conversation.append_tool_result(call_id, outcome.result)
        conversation.trim(self.max_turns)
        # The injected call travels with its outcome, so the confirmed
        # action appears in the inspector. Without it the one turn that
        # moved money rendered as 'no tools called'.
        return self._run(conversation, seeded=[outcome], seeded_calls=[injected])

    # --- internals -----------------------------------------------------------

    def _ensure_system(self, conversation: Conversation) -> None:
        if not conversation.messages or conversation.messages[0].get("role") != "system":
            conversation.messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    def _run(
        self,
        conversation: Conversation,
        seeded: list[ToolOutcome] | None = None,
        seeded_calls: list[ToolCall] | None = None,
    ) -> AgentResult:
        """Ask, execute, repeat -- until the model has words and wants nothing more."""
        calls: list[ToolCall] = list(seeded_calls or [])
        outcomes: list[ToolOutcome] = list(seeded or [])
        rounds = 0
        reply = Reply()

        for _ in range(self.max_tool_rounds):
            rounds += 1
            try:
                reply = self.client.complete(list(conversation.messages), list(TOOL_SCHEMAS))
            except LLMUnavailable as exc:
                # Carry what already ran. A transfer that executed before the
                # endpoint died still has to appear in the audit log; losing
                # it would make `turns` wrong exactly when something went
                # wrong, which is when it is read.
                log.warning("llm unavailable on round %d: %s", rounds, exc)
                return self._result(UNAVAILABLE, calls, outcomes, rounds, hit=False)

            conversation.append_assistant(reply)

            if not reply.tool_calls:
                break

            for call in reply.tool_calls:
                outcome = self.executor.execute(call)
                calls.append(call)
                outcomes.append(outcome)
                conversation.remember(outcome.sourced_cents, self.executor.writes)
                conversation.append_tool_result(call.id, outcome.result)

        conversation.trim(self.max_turns)

        spoken = (reply.text or "").strip()
        if not spoken:
            # No words and no rounds left: say something rather than nothing.
            return self._result(NOTHING_SAID, calls, outcomes, rounds, hit=False)

        sourced = _sourced(outcomes)
        # The customer's own figures are speakable. Repeating "$400" back to
        # someone who just said "transfer $400" is quoting them, not inventing
        # a fact -- and refusing to left the bot unable to take an instruction
        # at all, since every confirmation restated the amount.
        allowed = (
            sourced
            + conversation.recall(self.executor.writes)
            + _quoted(conversation)
            + _pending(self.executor)
        )
        checked, hit = guard(
            spoken,
            [o.result for o in outcomes],
            fallback_sentence(sourced),
            sourced_cents=allowed,
        )
        # Amounts go out in figures. The model writes them as words because
        # its replies are spoken, but "$3,000.00" is easier to read and far
        # easier to check at a glance -- and a speech synthesiser reads a
        # currency figure back as words anyway, so nothing is lost aloud.
        # Done after the guard, so the check runs on what the model actually
        # wrote.
        return self._result(
            in_digits(checked),
            calls,
            outcomes,
            rounds,
            hit=hit,
            suppressed=spoken if hit else None,
        )

    def _result(
        self,
        text: str,
        calls: list[ToolCall],
        outcomes: list[ToolOutcome],
        rounds: int,
        hit: bool,
        suppressed: str | None = None,
    ) -> AgentResult:
        return AgentResult(
            text=text,
            tool_calls=tuple(calls),
            tool_results=tuple(o.result for o in outcomes),
            sourced_cents=_sourced(outcomes),
            guard_hit=hit,
            suppressed=suppressed,
            rounds=rounds,
        )


def _sourced(outcomes: Sequence[ToolOutcome]) -> tuple[int, ...]:
    """Every figure the bank produced this turn, and nothing else."""
    return tuple(cents for outcome in outcomes for cents in outcome.sourced_cents)


def _quoted(conversation: Conversation, turns: int = 4) -> tuple[int, ...]:
    """Money figures the customer themselves stated recently, as cents.

    Only from `user` messages. The model's own earlier replies are not a
    source -- treating them as one would let an invented figure launder itself
    by being repeated.
    """
    said = [m for m in conversation.messages if m.get("role") == "user"][-turns:]
    return tuple(
        int(value * 100)
        for message in said
        for value in money_in_text(str(message.get("content") or ""))
    )


def _pending(executor: ToolExecutor) -> tuple[int, ...]:
    """The amount of an action that has been read back and not yet settled."""
    pending = executor.pending
    if pending is None:
        return ()
    amount = pending.arguments.get("amount_cents")
    return (amount,) if isinstance(amount, int) else ()
