"""An OpenAI-compatible chat client, over the standard library.

Written against a vLLM server, whose responses differ from the specification
in one way that matters: on a text reply the `tool_calls` key is **absent**,
not present and null. Every read of it goes through `.get`.

No HTTP dependency. `urllib.request` does everything needed here, and a
library for one POST is a library to keep up to date for one POST.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Any

from voicebot.llm.base import LLMClient, LLMUnavailable, Reply, ToolCall
from voicebot.utils.logging import get_logger

log = get_logger("llm.openai")

# Everything up to the last closing tag, whether or not an opening tag is
# there to match it.
_THINKING = re.compile(r"^.*</think>\s*", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove the model's reasoning, keeping only the answer after it.

    The server usually returns `reasoning: null` and clean content. Now and
    then, after a tool call, qwen3.6-35b-a3b-fp8 sends the answer, a closing
    </think>, and the answer again: the template opens the block itself, so
    no opening tag ever arrives, and a pattern that waited for one let the
    whole thing be read aloud. The cost of being wrong here is the bot
    speaking its own reasoning, and the check is one regex.
    """
    return _THINKING.sub("", text).strip() if text else text


def _post(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float) -> Any:
    """The only function here that touches the network, so tests have one seam."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class OpenAIClient(LLMClient):
    """Talks to a chat-completions endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 30.0,
        max_retries: int = 2,
        temperature: float = 0.4,
        max_tokens: int = 200,
        backoff_s: float = 0.5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.backoff_s = backoff_s

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
    ) -> Reply:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        # An empty tools list makes some servers refuse the request outright,
        # so the key is omitted rather than sent empty.
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        url = f"{self.base_url}/chat/completions"

        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._parse(_post(url, payload, headers, self.timeout_s))
            except urllib.error.HTTPError as exc:
                # A 4xx is our mistake and will fail again identically; only
                # retry the server's problems.
                if exc.code < 500:
                    raise LLMUnavailable(f"{url} rejected the request: {exc.code}") from exc
                last = exc
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                last = exc

            if attempt < self.max_retries:
                log.warning("llm attempt %d failed (%s); retrying", attempt + 1, last)
                time.sleep(self.backoff_s * (attempt + 1))

        raise LLMUnavailable(f"could not reach {url} after {self.max_retries + 1} attempts: {last}")

    @staticmethod
    def _parse(body: dict[str, Any]) -> Reply:
        """Read one chat-completions response.

        Static so the response shape is testable without a server.
        """
        choices = body.get("choices") or []
        if not choices:
            return Reply()
        message = choices[0].get("message") or {}

        content = message.get("content")
        text = strip_thinking(content) if content else None

        calls: list[ToolCall] = []
        for raw in message.get("tool_calls") or ():
            function = raw.get("function") or {}
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    # The executor rejects an empty argument set with a hint,
                    # which the model can act on. Raising here could not.
                    log.warning("undecodable tool arguments: %r", arguments)
                    arguments = {}
            calls.append(
                ToolCall(
                    id=raw.get("id", ""),
                    name=function.get("name", ""),
                    arguments=arguments if isinstance(arguments, dict) else {},
                )
            )

        return Reply(text=text, tool_calls=tuple(calls))
