"""The LLM stage interface and a dependency-free fake.

The fake is what keeps the test suite honest: the whole agent loop, including
tool calling and the confirmation gate, is exercised offline with no endpoint
and no network. An integration that cannot be tested without a server has cost
more than it added.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from voicebot.errors import ErrorCode, ProviderUnavailable


class LLMUnavailable(ProviderUnavailable):
    """The model could not be reached, or refused to answer.

    Raised only after retries are exhausted. The agent turns this into a
    sentence the user hears, never a 500. `code` says which way it failed: a
    timeout, a 429, a refused request, a server error.
    """

    default_component = "llm"


class ContextOverflow(LLMUnavailable):
    """The conversation is longer than the model can take.

    Sending it again unchanged fails again, so it is not retried as it is:
    the agent trims the history and asks once more.
    """

    default_code = ErrorCode.INVALID_INPUT


@dataclass(frozen=True)
class ToolCall:
    """One tool the model wants run, with the arguments it chose."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Reply:
    """One response from the model: words, tool calls, or both.

    `text` is None when the model only wants to call tools. Callers must cope
    with both being empty -- a model occasionally returns neither.
    """

    text: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()


class LLMClient(ABC):
    """Sends a conversation to a model and returns what it said."""

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
    ) -> Reply:
        """One round trip. Raises LLMUnavailable if the model cannot be reached.

        `tool_choice` is "auto" normally, or "required" to compel a call. The
        agent uses the second when the model has announced an action without
        taking it, where asking again has already been shown not to work.
        """


def _require_a_customer(messages: Sequence[dict[str, Any]]) -> None:
    """Refuse a history nobody is in, the way the real server does.

    An OpenAI-compatible server answers a conversation holding a tool call
    and its result but no user turn with 400, "no user query found". The fake
    used to answer it happily, so a bug that broke every card action in the
    browser passed the entire suite. Raising here turns that into a test
    failure, for the same reason running out of scripted replies does.
    """
    said = [m for m in messages if m.get("role") != "system"]
    if said and not any(m.get("role") == "user" for m in said):
        raise ValueError(
            "this history holds no user turn; a real server answers that "
            "with 400 rather than a reply"
        )


class FakeLLMClient(LLMClient):
    """Replays a scripted sequence of replies. Used by tests and CI."""

    def __init__(self, scripted: Sequence[Reply]) -> None:
        self._scripted = list(scripted)
        self._index = 0
        self.calls: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
        # What was asked of each round, so a test can assert that the retry
        # compelled a call rather than merely asking for one again.
        self.choices: list[str] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
    ) -> Reply:
        _require_a_customer(messages)
        self.calls.append((messages, tools))
        self.choices.append(tool_choice)
        if self._index >= len(self._scripted):
            raise IndexError(
                f"FakeLLMClient ran out of scripted replies after {self._index}; "
                "the test asked for one more than it supplied."
            )
        reply = self._scripted[self._index]
        self._index += 1
        return reply
