"""One conversational turn: ask the model, run what it asks for, check the answer.

The loop is bounded. A model that keeps calling tools is a model that is
stuck, and an unbounded loop against a shared endpoint is an outage rather
than a bug.

Nothing in here raises at its caller. A turn always produces something sayable,
because the alternative is a customer listening to silence.
"""

from __future__ import annotations

import json
import re
import secrets
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from voicebot.agent.executor import ToolExecutor, ToolOutcome
from voicebot.agent.guard import (
    allowed_values,
    fallback_sentence,
    guard,
    in_digits,
    money_in_text,
)
from voicebot.errors import ProviderTimeout, ToolError
from voicebot.llm.base import ContextOverflow, LLMClient, LLMUnavailable, Reply, ToolCall
from voicebot.llm.prompts import (
    HANDED_OVER,
    LANGUAGE_NOTE,
    NOTHING_SAID,
    SYSTEM_PROMPT,
    in_language,
)
from voicebot.llm.tools import TOOL_SCHEMAS
from voicebot.messages import say
from voicebot.utils.logging import get_logger
from voicebot.utils.observability import report
from voicebot.utils.resilience import checkpoint, current_deadline, deadline_context

log = get_logger("agent.loop")

DEFAULT_MAX_TOOL_ROUNDS = 4
DEFAULT_MAX_TURNS = 20

# How many turns a figure the bank gave stays quotable without a fresh lookup.
RECALL_TURNS = 6

# What an out-of-date tool result is replaced with. JSON, because that is what
# the model expects in a tool message.
STALE = '{"superseded": true, "note": "the account changed after this; look it up again"}'


# Things the model says when it is about to act: "let me check", "I'll pull
# up", "I'm looking that up", "one moment". Observed against the live model,
# which says them and then calls nothing at all.
#
# The verb has to follow the subject, so "which account would you like me to
# check?" does not match, and the stems are open-ended so "checking" and
# "looking" match the same entry as "check".
_ACTION = r"check|look|pull|grab|fetch|get|retriev|review|see|find|confirm|verify"
_ANNOUNCEMENT = re.compile(
    r"\b(?:"
    r"(?:let\s+me|i'?ll|i\s+will|i'?m|i\s+am|i\s+can|i\s+could)"
    r"\s+(?:just\s+|quickly\s+|go\s+ahead\s+and\s+)?(?:" + _ACTION + r")\w*"
    r"|one\s+moment|just\s+a\s+(?:moment|second)|hold\s+on|bear\s+with\s+me"
    r")\b",
    re.IGNORECASE,
)

# The same promises in Urdu, in the first person: "دیکھتا ہوں" and "دیکھ لیتا
# ہوں" (I'll take a look), "چیک کر رہا ہوں" (I'm checking), "معلوم کروں گا"
# (I'll find out) -- and "انتظار کریں", "ایک لمحہ": please wait, one moment.
# Observed live as "...موجود رقم دیکھتا ہوں۔ براہ کرم انتظار کریں۔" with
# nothing called. The ending is what makes it the bot speaking, so "آپ دیکھ
# سکتے ہیں" -- you can see -- is not a promise.
_ACTION_UR = r"(?:دیکھ|چیک\s+کر|معلوم\s+کر|پتہ\s+کر|تلاش\s+کر|نکال)"
_ANNOUNCEMENT_UR = re.compile(
    _ACTION_UR
    + r"(?:(?:تا|تی)\s+ہوں|\s+(?:رہا|رہی|لیتا|لیتی)\s+ہوں|(?:وں|\s+لوں)\s+(?:گا|گی))"
    + r"|انتظار\s+(?:کریں|کیجیے|کیجئے|فرمائیں)|ایک\s+لمحہ|ذرا\s+رک"
)


def announces_action(text: str | None) -> bool:
    """True when a reply promises to look something up instead of doing it.

    A reply that ends in a question is left alone whatever else it says. The
    bot is meant to ask which account before fetching a balance, and phrasing
    that as "I can check that for you -- which one?" must stay a question
    rather than be turned into a lookup of the wrong account.

    This is a guess about English and Urdu, and it is allowed to be: a miss
    leaves the old behaviour, and a false positive costs one extra round.
    What follows it is not a guess.
    """
    said = (text or "").strip()
    # "؟" is the question mark Urdu is written with.
    if not said or said.endswith(("?", "؟")):
        return False
    return bool(_ANNOUNCEMENT.search(said) or _ANNOUNCEMENT_UR.search(said))


# Asking the customer to agree to something: "are you sure", "shall I",
# "would you like me to". Legitimate when the gate is holding an action and
# broken when it is not, which is why the wording alone never decides.
_AGREEMENT = re.compile(
    r"ARE\s+YOU\s+SURE|SHALL\s+I|SHOULD\s+I|DO\s+YOU\s+WANT\s+ME\s+TO"
    r"|WOULD\s+YOU\s+LIKE\s+ME\s+TO|GO\s+AHEAD|PROCEED|CONFIRM",
    re.IGNORECASE,
)


def seeks_agreement(text: str | None, armed: bool) -> bool:
    """True when a reply asks for a yes that nothing is waiting to receive.

    `armed` is the whole of the safety here, and it is a fact rather than a
    reading: the executor either holds a pending action or it does not. When
    it does, a read-back is the gate working and is left exactly alone.

    The amount is required so that an offer with no figure in it -- "which
    account would you like me to check?" -- is not mistaken for a read-back.
    """
    if armed:
        return False
    said = (text or "").strip()
    if not said.endswith("?") or not _AGREEMENT.search(said):
        return False
    return bool(money_in_text(said))


# Asking for a human, in the ways people actually ask. Matched against what
# the customer said, never against what the model replied.
_WANTS_A_PERSON = re.compile(
    r"(?:speak|talk|put\s+me\s+through|transfer\s+me|get\s+me)"
    r"(?:\s+\w+){0,3}?\s+"
    r"(?:a\s+|an\s+|the\s+)?"
    r"(?:human|person|someone|somebody|agent|representative|advis[eo]r|operator|staff)"
    r"|real\s+(?:human|person)"
    r"|customer\s+(?:service|support)"
    r"|human\s+being",
    re.IGNORECASE,
)

# The same request in Urdu: a person to talk to ("کسی انسان سے بات کرنی ہے",
# "نمائندے سے بات کروائیں"), or customer service. "Talk" has to follow the
# person, because a case is opened on a match: "انسان سے غلطی ہو جاتی ہے" --
# people make mistakes -- is not a request for one.
_WANTS_A_PERSON_UR = re.compile(
    r"(?:انسان|نمائندے|نمائندہ|ایجنٹ|بندے|شخص|آپریٹر|عملے)\s+سے\s+(?:\S+\s+){0,2}?بات"
    r"|کسٹمر\s+(?:سروس|سپورٹ|کیئر)"
)

# And as Urdu is typed, in Latin letters: "kisi insaan se baat karni hai",
# "agent se baat karao". "insaan se ghalti ho jati hai" is not a request.
_WANTS_A_PERSON_ROMAN = re.compile(
    r"\b(?:insaan|insan|banda|bande|banday|numainda|numainde|numainday|agent|operator|staff)"
    r"\s+se\s+(?:\w+\s+){0,2}?baat\b",
    re.IGNORECASE,
)


def asked_for_a_person(text: str | None) -> bool:
    """True when the customer asked to be put through to a human.

    Reading a language is a guess, and this one is allowed to be: a miss
    leaves the model's own judgement, which the prompt already directs. What
    follows a match is not a guess -- the case is opened in Python either way.
    """
    said = (text or "").strip()
    return any(
        pattern.search(said)
        for pattern in (_WANTS_A_PERSON, _WANTS_A_PERSON_UR, _WANTS_A_PERSON_ROMAN)
    )


@dataclass
class Conversation:
    """One caller's message history, in the shape the API expects."""

    session_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    turn_index: int = 0
    # The language of the customer's latest turn, "en" or "ur". The model
    # follows the customer by itself; this is for the sentences the bot says
    # without it.
    language: str = "en"
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

    def invalidate_tool_results(self) -> None:
        """Replace earlier tool results, because the account just changed.

        A bill list fetched before a payment still says the bill is unpaid,
        and a model reading its own context will answer from it. Telling it to
        look again is advice it follows most of the time, which is the wrong
        kind of guarantee for a bank. Removing the values is not advice.

        The result of the change itself is appended after this runs, so it
        survives -- it is the only one that is still true.
        """
        for message in self.messages:
            if message.get("role") == "tool":
                message["content"] = STALE

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


def with_language_note(conversation: Conversation) -> list[dict[str, Any]]:
    """The history as the model is sent it: the customer's latest message
    carries the language to answer in.

    Next to the message is the one place the note was measured to work. At
    the end of the system prompt it was ignored as often as followed, and a
    system message nearer the reply is refused by the server ("System message
    must be at the beginning"). Only this copy carries it: the history keeps
    the customer's words exactly as they were said.
    """
    messages = list(conversation.messages)
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            note = LANGUAGE_NOTE.get(conversation.language, LANGUAGE_NOTE["en"])
            said = messages[index].get("content") or ""
            messages[index] = {**messages[index], "content": f"{said}\n\n({note})"}
            break
    return messages


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
    # The reference of a case this turn opened, however it was opened. The
    # page shows it; a caller can tell a handed-off turn from an ordinary one
    # without reading the reply.
    handed_off: str | None = None
    # The reference of a dispute this turn filed. Separate from a handoff:
    # one means the bot stopped, the other that a claim was raised and the
    # conversation carries on.
    dispute_filed: str | None = None
    # The card and purpose a PIN was asked for, if one was. The page opens
    # its keypad from this; nothing in the turn ever holds the PIN itself.
    pin_request: dict[str, str] | None = None
    # When the turn ended in one of the bot's own fallback sentences rather
    # than an answer, which one (a key in messages.yaml). The conversation
    # counts these: several in a row and a colleague is asked to take over.
    fallback: str | None = None


# A model round is not started with less than this left of the turn's budget:
# it would only be cut off, and the customer would hear nothing for it.
MIN_ROUND_S = 1.0


class Agent:
    """Drives one conversation against the model and its tools."""

    def __init__(
        self,
        client: LLMClient,
        executor: ToolExecutor,
        max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
        max_turns: int = DEFAULT_MAX_TURNS,
        turn_budget_s: float = 20.0,
    ) -> None:
        self.client = client
        self.executor = executor
        self.max_tool_rounds = max_tool_rounds
        self.max_turns = max_turns
        # Every model round of one turn, together. See voicebot.utils.resilience.
        self.turn_budget_s = turn_budget_s

    # --- public --------------------------------------------------------------

    def handle(self, text: str, conversation: Conversation) -> AgentResult:
        """Advance the conversation by one turn."""
        self._ensure_system(conversation)
        conversation.turn_index += 1
        # The executor opens handoffs and has to say which turn raised one.
        # It never advances this itself; the conversation owns the count.
        self.executor.turn_index = conversation.turn_index
        conversation.append_user(text)
        conversation.trim(self.max_turns)

        with deadline_context(self.turn_budget_s):
            result = self._run(conversation)
        if result.handed_off is None and asked_for_a_person(text):
            result = self._hand_off_by_rule(conversation, result)
        return result

    def resolve_pending(self, confirmed: bool, conversation: Conversation) -> AgentResult:
        """Act on the Yes or Cancel button, then let the model say what happened.

        The result is appended to the history deliberately. Skipping it leaves
        a model whose context still says it is waiting for an answer, so the
        next "did that go through?" is answered from a history that is wrong.
        """
        self._ensure_system(conversation)
        self.executor.turn_index = conversation.turn_index
        spoken_to = any(m.get("role") == "user" for m in conversation.messages)
        pending = self.executor.pending
        before = self.executor.writes
        outcome = self.executor.confirm() if confirmed else self.executor.cancel()

        if outcome.failed is not None and pending is not None:
            # The bank failed under the action itself. What to say is known --
            # nothing was done, or it is not known whether it was -- and a
            # model asked to phrase it is a model asked to guess.
            call_id = f"confirm-{secrets.token_urlsafe(6)}"
            failed_call = ToolCall(id=call_id, name=pending.tool, arguments=dict(pending.arguments))
            if spoken_to:
                conversation.append_assistant(Reply(tool_calls=(failed_call,)))
                conversation.append_tool_result(call_id, outcome.result)
            return self._fallback(outcome.failed, conversation, [failed_call], [outcome], rounds=0)

        if outcome.result.get("status") == "pin_required":
            # Yes was given, the PIN has not been. Nothing ran and the action
            # is still pending, so the history is left exactly as it was: an
            # injected tool call here would tell the model the money moved,
            # and its next answer would say so.
            #
            # The model is not asked to phrase this either. It has nothing to
            # add to "enter your PIN", and a round trip to hear that is time
            # the customer spends looking at a keypad that has not appeared.
            log.info("confirmed, now holding for a PIN")
            return self._result(
                in_language(str(outcome.result.get("detail", "")), conversation.language),
                calls=[],
                outcomes=[outcome],
                rounds=0,
                hit=False,
            )

        if self.executor.writes != before:
            conversation.invalidate_tool_results()

        # A real tool name and a fresh id: duplicate ids across a session, or
        # a name absent from the schemas, are the kind of thing a stricter
        # server rejects -- on the one path that moves money.
        call_id = f"confirm-{secrets.token_urlsafe(6)}"
        injected = ToolCall(
            id=call_id,
            name=pending.tool if pending else "pay_bill",
            arguments=dict(pending.arguments) if pending else {},
        )
        calls = [injected]
        conversation.append_assistant(Reply(tool_calls=(injected,)))
        conversation.remember(outcome.sourced_cents, self.executor.writes)
        conversation.append_tool_result(call_id, outcome.result)
        conversation.trim(self.max_turns)
        if not spoken_to:
            # Nobody has said anything in this conversation, so there is
            # nothing for the model to answer and no history worth sending:
            # a real server refuses one holding a tool call and its result
            # and no customer. Happens when Yes is clicked twice, or when a
            # second tab confirms a transfer armed in the first.
            said = outcome.result.get("detail") or outcome.result.get("error")
            log.info("settling without the model: nothing was ever said here")
            return self._result(
                in_language(str(said or NOTHING_SAID), conversation.language),
                calls,
                [outcome],
                rounds=0,
                hit=False,
            )

        # The injected call travels with its outcome, so the confirmed
        # action appears in the inspector. Without it the one turn that
        # moved money rendered as 'no tools called'.
        with deadline_context(self.turn_budget_s):
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
        choice = "auto"
        trimmed = False

        while rounds < self.max_tool_rounds:
            # Stop here, between rounds, if the caller has gone: the history
            # is whole at this point, so nothing is left half-written.
            checkpoint("model round")
            deadline = current_deadline()
            if deadline is not None and deadline.remaining() < MIN_ROUND_S:
                report(
                    ProviderTimeout(
                        f"the turn's {deadline.seconds:.0f}s budget ran out "
                        f"after {rounds} round(s)",
                        component="agent",
                    )
                )
                return self._unavailable(conversation, calls, outcomes, rounds)
            rounds += 1
            try:
                reply = self.client.complete(
                    with_language_note(conversation), list(TOOL_SCHEMAS), choice
                )
            except ContextOverflow:
                if trimmed:
                    return self._unavailable(conversation, calls, outcomes, rounds)
                # Too long for the model. The oldest turns go -- the latest
                # two are kept, with whatever this turn has done so far -- and
                # the round is asked once more.
                log.warning("the conversation is too long for the model; trimming it")
                conversation.trim(2)
                trimmed = True
                rounds -= 1
                continue
            except LLMUnavailable:
                # Reported where it failed, with its code and the call. What
                # already ran is carried: a transfer that executed before the
                # endpoint died still has to appear in the audit log, and the
                # customer has to hear that it went through.
                return self._unavailable(conversation, calls, outcomes, rounds)

            conversation.append_assistant(reply)

            if not reply.tool_calls:
                # The model said it would look something up and then called
                # nothing. Left alone the turn ends on the promise, and the
                # customer is told the bot is checking by a bot that has
                # stopped. Worse, the promise stays in the history, and the
                # next round answers from a model that has now invented the
                # figures it never fetched.
                #
                # Asking again changes nothing -- it is what already failed --
                # so the retry compels the call instead of requesting it.
                if not calls and choice == "auto" and self._owes_a_lookup(reply.text, conversation):
                    log.info("forcing a tool call: the model promised one and made none")
                    # Struck from the history, not just replaced. The customer
                    # never heard it -- the turn returns one reply -- and a
                    # model that finds its own "are you sure?" above a tool
                    # result answers itself. Every forced deposit came back
                    # as "Yes" until this line.
                    conversation.messages.pop()
                    choice = "required"
                    continue
                break

            choice = "auto"
            failed: str | None = None
            for call in reply.tool_calls:
                if failed is not None:
                    # Every call gets a result, or the next request is a
                    # history with a dangling call in it, which the server
                    # refuses -- every turn, until it scrolls away.
                    conversation.append_tool_result(
                        call.id,
                        {"error": "Not run: an earlier tool failed.", "hint": "Do not retry it."},
                    )
                    continue
                before = self.executor.writes
                outcome = self.executor.execute(call)
                if self.executor.writes != before:
                    conversation.invalidate_tool_results()
                calls.append(call)
                outcomes.append(outcome)
                conversation.remember(outcome.sourced_cents, self.executor.writes)
                conversation.append_tool_result(call.id, outcome.result)
                failed = outcome.failed
            if failed is not None:
                # The bank failed under a tool. The model is not asked what to
                # make of that: told a lookup failed, it fills the gap with a
                # guess. The bot says what it knows, and that is all.
                return self._fallback(failed, conversation, calls, outcomes, rounds, record=True)

        conversation.trim(self.max_turns)

        spoken = (reply.text or "").strip()
        pending = self.executor.pending
        if (
            not spoken
            and pending is not None
            and any(o.result.get("status") == "needs_confirmation" for o in outcomes)
        ):
            # The rounds ran out on the call that armed an action, before the
            # model could read it back. The bank's own summary is the
            # read-back, and it is the sentence the Yes button acts on.
            log.warning("no words after arming %s; reading back its summary", pending.tool)
            return self._result(f"{pending.summary}?", calls, outcomes, rounds, hit=False)
        if not spoken:
            # No words and no rounds left: say something rather than nothing.
            return self._fallback("nothing_said", conversation, calls, outcomes, rounds)

        if not calls and self._unfinished(spoken):
            # Compelled to call a tool, it still did not. Whatever it wrote is
            # a promise nothing will keep, so it is not worth saying. Asking
            # the customer to try again is at least true.
            log.warning("suppressing an unkept promise")
            log.debug("the promise was: %r", spoken)
            return self._fallback(
                "not_looked_up", conversation, calls, outcomes, rounds, suppressed=spoken
            )

        sourced = _sourced(outcomes)
        allowed = self._allowed(conversation, sourced)
        checked, hit = guard(
            spoken,
            [o.result for o in outcomes],
            fallback_sentence(sourced, conversation.language),
            sourced_cents=allowed,
        )
        # Amounts go out in figures, including the ones the model wrote with
        # no "dollars" attached -- `allowed` is what identifies those, so the
        # rewrite never guesses. The model writes them as words because
        # its replies are spoken, but "$3,000.00" is easier to read and far
        # easier to check at a glance -- and a speech synthesiser reads a
        # currency figure back as words anyway, so nothing is lost aloud.
        # Done after the guard, so the check runs on what the model actually
        # wrote.
        if hit and self.executor.pending is not None:
            # The sentence the customer was meant to agree to never reached
            # them, so there is nothing for them to agree to. Leaving the
            # action armed let a "yes" aimed at something else pay a bill
            # nobody had heard named.
            log.warning("cancelling %s: its read-back was suppressed", self.executor.pending.tool)
            self.executor.cancel()

        return self._result(
            in_digits(checked, allowed),
            calls,
            outcomes,
            rounds,
            hit=hit,
            suppressed=spoken if hit else None,
            # A guard hit with no real figure to read instead is not an
            # answer; one that reads the bank's figure back is.
            fallback=(
                "unverified"
                if hit and checked == say("unverified", conversation.language)
                else None
            ),
        )

    def _unavailable(
        self,
        conversation: Conversation,
        calls: list[ToolCall],
        outcomes: list[ToolOutcome],
        rounds: int,
    ) -> AgentResult:
        """The model cannot be reached, or the turn ran out of time: say what is true.

        If something already happened this turn, that is what the customer
        must hear. "I can't reach the service, please try again" after Yes has
        moved the money invites them to move it a second time.
        """
        statuses = [o.result.get("status") for o in outcomes]
        if "done" in statuses:
            return self._fallback("action_done", conversation, calls, outcomes, rounds)
        if "cancelled" in statuses:
            return self._fallback("action_cancelled", conversation, calls, outcomes, rounds)
        pending = self.executor.pending
        if pending is not None and "needs_confirmation" in statuses:
            # Armed this turn: its read-back is the bank's summary, and the
            # Yes button can still act on it without the model.
            return self._result(f"{pending.summary}?", calls, outcomes, rounds, hit=False)
        return self._fallback("unavailable", conversation, calls, outcomes, rounds)

    def _fallback(
        self,
        key: str,
        conversation: Conversation,
        calls: list[ToolCall],
        outcomes: list[ToolOutcome],
        rounds: int,
        suppressed: str | None = None,
        record: bool = False,
        **fields: str,
    ) -> AgentResult:
        """End the turn with one of the bot's own sentences, in the customer's language.

        `record` writes it into the history as the bot's reply, so the next
        round -- which will see the failed tool's result -- also sees what the
        customer was told about it.
        """
        text = say(key, conversation.language, **fields)
        if record:
            conversation.append_assistant(Reply(text=text))
            conversation.trim(self.max_turns)
        return self._result(
            text, calls, outcomes, rounds, hit=False, suppressed=suppressed, fallback=key
        )

    def _hand_off_by_rule(self, conversation: Conversation, result: AgentResult) -> AgentResult:
        """Open a case because the customer asked, whatever the model replied.

        The model is told to do this and usually will. When it does not --
        and a model that has just been asked to step aside is exactly the one
        inclined to argue -- the rule does it anyway. Enforced here for the
        same reason the confirmation gate is enforced in Python: a model can
        be talked out of an instruction and a function cannot.

        The sentence spoken is assembled from the case, not written by the
        model, because the reference is the one part that has to be right.
        """
        call = ToolCall(
            id=f"handoff-{secrets.token_urlsafe(6)}",
            name="hand_off",
            arguments={"reason": "customer asked"},
        )
        outcome = self.executor.execute(call)
        reference = _reference([outcome], "handed_off")
        if reference is None:
            # They asked for a person and no case could be opened -- the bank
            # is down, most likely. The model's reply would carry on as if
            # nothing had been asked; the customer needs another way through.
            report(
                ToolError("could not open the handoff the customer asked for", component="hand_off")
            )
            return self._fallback(
                "cannot_complete",
                conversation,
                [*result.tool_calls, call],
                [ToolOutcome(r) for r in result.tool_results] + [outcome],
                result.rounds,
            )

        # Recorded in the history too, so the next turn knows the conversation
        # has already left the bot.
        conversation.append_assistant(Reply(tool_calls=(call,)))
        conversation.append_tool_result(call.id, outcome.result)
        conversation.trim(self.max_turns)
        log.info("handing off by rule: the customer asked for a person")

        return AgentResult(
            text=in_language(HANDED_OVER, conversation.language).format(reference=reference),
            tool_calls=(*result.tool_calls, call),
            tool_results=(*result.tool_results, outcome.result),
            sourced_cents=result.sourced_cents,
            guard_hit=result.guard_hit,
            suppressed=result.suppressed,
            rounds=result.rounds,
            handed_off=reference,
            dispute_filed=result.dispute_filed,
            pin_request=result.pin_request,
        )

    def _allowed(self, conversation: Conversation, sourced: tuple[int, ...]) -> tuple[int, ...]:
        """Every figure this turn is entitled to say.

        The customer's own figures are speakable. Repeating "$400" back to
        someone who just said "transfer $400" is quoting them, not inventing
        a fact -- and refusing left the bot unable to take an instruction at
        all, since every confirmation restated the amount.
        """
        return (
            sourced
            + conversation.recall(self.executor.writes)
            + _quoted(conversation)
            + _pending(self.executor)
        )

    def _unfinished(self, text: str | None) -> bool:
        """True when the reply leaves the turn owing the customer an action.

        Two shapes of the same failure: announcing a lookup and not making
        one, and asking for a yes with nothing armed to receive it. Both end
        a turn having told the customer something is happening when nothing
        is.
        """
        return announces_action(text) or seeks_agreement(
            text, armed=self.executor.pending is not None
        )

    def _owes_a_lookup(self, text: str | None, conversation: Conversation) -> bool:
        """True when the turn cannot honestly end here.

        The two unfinished shapes above, plus a third found live: a figure
        stated outright with nothing looked up to state it from. The guard
        stops that reaching the customer, but stopping a wrong answer is not
        the same as giving a right one, and the right one is a lookup away.

        Deliberately not used to decide what is finally spoken -- that stays
        the guard's, so an invention is still counted as one.
        """
        if self._unfinished(text):
            return True
        allowed = allowed_values(self._allowed(conversation, ()))
        return any(value not in allowed for value in money_in_text(text or ""))

    def _result(
        self,
        text: str,
        calls: list[ToolCall],
        outcomes: list[ToolOutcome],
        rounds: int,
        hit: bool,
        suppressed: str | None = None,
        fallback: str | None = None,
    ) -> AgentResult:
        return AgentResult(
            text=text,
            tool_calls=tuple(calls),
            tool_results=tuple(o.result for o in outcomes),
            sourced_cents=_sourced(outcomes),
            guard_hit=hit,
            suppressed=suppressed,
            rounds=rounds,
            handed_off=_reference(outcomes, "handed_off"),
            dispute_filed=_reference(outcomes, "filed"),
            pin_request=_pin_request(outcomes),
            fallback=fallback,
        )


def _pin_request(outcomes: Sequence[ToolOutcome]) -> dict[str, str] | None:
    """The keypad this turn asked for, if it asked for one.

    Two things ask: the model, through `ask_for_pin`, and the gate in front
    of money leaving an account. The page opens the same keypad for both.
    """
    for outcome in outcomes:
        if outcome.result.get("status") in {"pin_requested", "pin_required"}:
            asked = {
                "last4": str(outcome.result.get("last4", "")),
                "purpose": str(outcome.result.get("purpose", "")),
            }
            # What the PIN approves, so the keypad can say it. The dialogue
            # covers the confirmation bar, and a customer should not be
            # typing a PIN against something they can no longer read.
            summary = outcome.result.get("summary")
            if summary:
                asked["summary"] = str(summary)
            return asked
    return None


def _reference(outcomes: Sequence[ToolOutcome], status: str) -> str | None:
    """The reference of a case this turn opened with that status, if any."""
    for outcome in outcomes:
        if outcome.result.get("status") == status:
            reference = outcome.result.get("reference")
            if isinstance(reference, str):
                return reference
    return None


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
