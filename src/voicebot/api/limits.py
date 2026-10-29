"""How long a conversation may run, and what happens when its turns keep failing.

A turn that ends in one of the bot's own fallback sentences -- not heard, not
answered, a lookup failed -- is a failed turn, and so is the same reply given
twice in a row: the customer is asking and the bot is not getting it. A few
of those in a row and the conversation is handed to a person, with a
reference, rather than left to fail a fourth time; when even that cannot be
done, the customer is told how else to get help.

A conversation also ends: after `max_turns` turns or `max_minutes` minutes,
with a warning a few turns before, and a polite close that starts it afresh.
Without that a stuck page, or a script, runs up cost against the model for
ever.
"""

from __future__ import annotations

import dataclasses
import secrets

from voicebot.agent.loop import Conversation
from voicebot.api.sessions import Session
from voicebot.config import SessionConfig
from voicebot.llm.base import Reply, ToolCall
from voicebot.messages import say
from voicebot.pipeline.orchestrator import TurnResult, VoicePipeline
from voicebot.utils.logging import get_logger
from voicebot.utils.observability import count

log = get_logger("api.limits")

# The fallbacks that mean the customer did not get what they came for.
FAILED: frozenset[str] = frozenset(
    {
        "unavailable",
        "nothing_said",
        "not_caught",
        "speech_unavailable",
        "bad_audio",
        "not_looked_up",
        "unverified",
        "tool_failed",
        "outcome_unknown",
        "internal",
    }
)


class ConversationLimits:
    """The rules above, applied around each turn."""

    def __init__(self, config: SessionConfig) -> None:
        self.config = config

    def over(self, session: Session) -> bool:
        """Has this conversation run its course?"""
        return (
            session.turns >= self.config.max_turns or session.minutes() >= self.config.max_minutes
        )

    def close(self, session: Session, pipeline: VoicePipeline) -> TurnResult:
        """Say goodbye, and start the conversation afresh for whoever speaks next."""
        language = session.conversation.language
        log.info("closing the conversation after %d turns", session.turns)
        count("session_ended")
        result = pipeline.say(say("session_ended", language), key="session_ended")
        session.conversation = Conversation(session.conversation.session_id)
        session.conversation.language = language
        session.executor.pending = None
        session.turns, session.failures, session.last_reply, session.warned = 0, 0, "", False
        session.started = _now()
        return result

    def settle(self, session: Session, pipeline: VoicePipeline, result: TurnResult) -> TurnResult:
        """After a turn: count it, hand over a run of failures, warn near the end."""
        session.turns += 1
        key = result.agent.fallback
        repeated = bool(result.agent.text) and result.agent.text == session.last_reply
        session.last_reply = result.agent.text

        if key in FAILED or (repeated and result.agent.handed_off is None):
            session.failures += 1
            log.info(
                "failed turn %d in a row (%s)",
                session.failures,
                key or "the same reply again",
            )
        elif key is None:
            session.failures = 0

        if session.failures >= self.config.failure_limit:
            return self._escalate(session, pipeline, result)

        near_end = session.turns >= self.config.max_turns - self.config.warn_turns or (
            session.minutes() >= self.config.max_minutes * 0.9
        )
        if near_end and not session.warned:
            session.warned = True
            return pipeline.say_after(result, say("session_warning", session.conversation.language))
        return result

    def _escalate(
        self, session: Session, pipeline: VoicePipeline, result: TurnResult
    ) -> TurnResult:
        """Several turns in a row have failed: ask a colleague to take over."""
        failures, session.failures = session.failures, 0
        conversation = session.conversation
        count("escalations")
        call = ToolCall(
            id=f"escalate-{secrets.token_urlsafe(6)}",
            name="hand_off",
            arguments={
                "reason": "stuck",
                "note": f"{failures} turns in a row failed; the bot handed over",
            },
        )
        outcome = session.executor.execute(call)
        reference = outcome.result.get("reference")
        if isinstance(reference, str) and outcome.result.get("status") == "handed_off":
            log.warning("handed over after %d failed turns", failures)
            text, key, handed_off = (
                say("escalated", conversation.language, reference=reference),
                "escalated",
                reference,
            )
        else:
            log.error("could not hand over after %d failed turns", failures)
            text, key, handed_off = (
                say("cannot_complete", conversation.language),
                "cannot_complete",
                None,
            )

        # Recorded, so the next turn knows the conversation has left the bot.
        if any(m.get("role") == "user" for m in conversation.messages):
            conversation.append_assistant(Reply(tool_calls=(call,)))
            conversation.append_tool_result(call.id, outcome.result)
            conversation.append_assistant(Reply(text=text))
        agent = dataclasses.replace(
            result.agent,
            text=text,
            fallback=key,
            handed_off=handed_off or result.agent.handed_off,
            tool_calls=(*result.agent.tool_calls, call),
            tool_results=(*result.agent.tool_results, outcome.result),
        )
        return pipeline.replace_reply(result, agent)


def _now() -> float:
    import time

    return time.monotonic()
