"""An OpenAI-compatible chat client, over the standard library.

Written against a vLLM server, whose responses differ from the specification
in one way that matters: on a text reply the `tool_calls` key is **absent**,
not present and null. Every read of it goes through `.get`.

No HTTP dependency. `urllib.request` does everything needed here, and a
library for one POST is a library to keep up to date for one POST.

Every failure becomes an LLMUnavailable carrying its kind -- timeout, 429,
5xx, refused, malformed -- and the request is made through call_with_retry,
so it is retried only when that can help, backs off with jitter, honours
Retry-After, stays inside the turn's deadline, and stops being tried at all
for a while once the server has failed several times in a row.
"""

from __future__ import annotations

import http.client
import json
import re
import socket
import time
import urllib.error
import urllib.request
from typing import Any

from voicebot.errors import ErrorCode, code_for_status, retry_after_seconds
from voicebot.llm.base import ContextOverflow, LLMClient, LLMUnavailable, Reply, ToolCall
from voicebot.utils.logging import get_logger
from voicebot.utils.resilience import RetryPolicy, breaker, call_with_retry

log = get_logger("llm.openai")

# Everything up to the last closing tag, whether or not an opening tag is
# there to match it.
_THINKING = re.compile(r"^.*</think>\s*", re.DOTALL)

# How a server says the conversation is longer than the model can take.
_OVERFLOW = re.compile(r"context length|context window|too many tokens|maximum.*tokens", re.I)

# The end of the last whole sentence in a reply the token limit cut short.
# A full stop straight after a digit, at the very end, is not one: it is the
# decimal point of an amount the limit cut in half -- measured, an Urdu list
# of bills ended "...$1,200." where it said "$1,200.00", and "$2,483.50"
# would have been read as "$2,483".
_SENTENCE_END = re.compile(r"(?:\.(?=\s)|(?<!\d)\.$|[!?؟۔](?=\s|$))")

_READ_CHUNK = 65_536


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
    """The only function here that touches the network, so tests have one seam.

    `timeout` bounds the whole exchange, not each socket read: a server that
    sends its answer a few bytes at a time would otherwise keep a caller
    waiting for far longer than the timeout says.
    """
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    finish = time.monotonic() + timeout
    with urllib.request.urlopen(request, timeout=timeout) as response:
        chunks: list[bytes] = []
        while True:
            if time.monotonic() > finish:
                raise TimeoutError(f"no complete answer within {timeout:.1f}s")
            chunk = response.read(_READ_CHUNK)
            if not chunk:
                break
            chunks.append(chunk)
    return json.loads(b"".join(chunks).decode("utf-8"))


def _http_error(exc: urllib.error.HTTPError, url: str) -> LLMUnavailable:
    """What an HTTP error status means, as an LLMUnavailable of the right kind."""
    try:
        detail = exc.read(300).decode("utf-8", "replace") if exc.fp else ""
    except Exception:  # the body is only for the log; failing to read it changes nothing
        detail = ""
    if exc.code == 400 and _OVERFLOW.search(detail):
        return ContextOverflow(f"{url} said the conversation is too long", cause=exc)
    retry_after = retry_after_seconds(exc.headers.get("Retry-After") if exc.headers else None)
    return LLMUnavailable(
        f"{url} answered {exc.code}: {detail[:200]}".rstrip(": "),
        code=code_for_status(exc.code),
        retry_after=retry_after,
        cause=exc,
    )


class OpenAIClient(LLMClient):
    """Talks to a chat-completions endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 12.0,
        max_retries: int = 1,
        temperature: float = 0.4,
        max_tokens: int = 200,
        backoff_s: float = 0.2,
        backoff_cap_s: float = 1.0,
        name: str = "llm",
        disable_thinking: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.disable_thinking = disable_thinking
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.name = name
        self.policy = RetryPolicy(attempts=max_retries + 1, base_s=backoff_s, cap_s=backoff_cap_s)

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
        if self.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        # An empty tools list makes some servers refuse the request outright,
        # so the key is omitted rather than sent empty.
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        url = f"{self.base_url}/chat/completions"

        def attempt(timeout: float) -> Reply:
            try:
                body = _post(url, payload, headers, timeout)
            except urllib.error.HTTPError as exc:
                raise _http_error(exc, url) from exc
            except TimeoutError as exc:
                raise LLMUnavailable(
                    f"{url} did not answer within {timeout:.1f}s",
                    code=ErrorCode.PROVIDER_TIMEOUT,
                    cause=exc,
                ) from exc
            except urllib.error.URLError as exc:
                timed_out = isinstance(exc.reason, TimeoutError | socket.timeout)
                raise LLMUnavailable(
                    f"could not reach {url}: {exc.reason}",
                    code=ErrorCode.PROVIDER_TIMEOUT if timed_out else None,
                    cause=exc,
                ) from exc
            except (http.client.HTTPException, ConnectionError, OSError) as exc:
                # A dropped connection, a reset, an answer cut off part-way.
                raise LLMUnavailable(f"the connection to {url} failed: {exc!r}", cause=exc) from exc
            except ValueError as exc:
                raise LLMUnavailable(f"{url} sent something that is not JSON", cause=exc) from exc
            return self._parse(body)

        return call_with_retry(
            attempt,
            component=self.name,
            timeout_s=self.timeout_s,
            policy=self.policy,
            circuit=breaker(self.name),
            error_type=LLMUnavailable,
        )

    @staticmethod
    def _parse(body: Any) -> Reply:
        """Read one chat-completions response.

        Static so the response shape is testable without a server. Anything
        that is not the shape expected is a malformed answer: retryable, since
        the next one may well be fine, and never an exception of another kind
        escaping into the turn.
        """
        try:
            return OpenAIClient._read(body)
        except (AttributeError, TypeError, KeyError, IndexError) as exc:
            raise LLMUnavailable(f"a malformed answer: {exc!r}", cause=exc) from exc

    @staticmethod
    def _read(body: Any) -> Reply:
        if not isinstance(body, dict):
            raise TypeError(f"the answer is a {type(body).__name__}, not an object")
        if isinstance(body.get("error"), dict | str):
            raise LLMUnavailable(f"the server answered with an error: {str(body['error'])[:200]}")

        choices = body.get("choices") or []
        if not choices:
            log.warning("the model answered with no choices at all")
            return Reply()
        choice = choices[0]
        message = choice.get("message") or {}

        content = message.get("content")
        if isinstance(content, list):
            # Some servers send content as parts; only the text parts are speech.
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        text = strip_thinking(content) if isinstance(content, str) and content else None

        if text and choice.get("finish_reason") == "length":
            # Cut off by the token limit. A half sentence read aloud is worse
            # than a shorter whole one, so it ends at the last full stop.
            ends = list(_SENTENCE_END.finditer(text))
            log.warning(
                "the reply was cut off by the token limit; %d whole sentences kept", len(ends)
            )
            text = text[: ends[-1].end()] if ends else text

        calls: list[ToolCall] = []
        for raw in message.get("tool_calls") or ():
            if not isinstance(raw, dict):
                log.warning("skipped a tool call that is not an object")
                continue
            function = raw.get("function") or {}
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments) if arguments.strip() else {}
                except json.JSONDecodeError:
                    # The executor rejects an empty argument set with a hint,
                    # which the model can act on. Raising here could not.
                    log.warning("undecodable arguments for %s", function.get("name"))
                    log.debug("the arguments were: %r", arguments)
                    arguments = {}
            calls.append(
                ToolCall(
                    id=str(raw.get("id", "")),
                    name=str(function.get("name", "")),
                    arguments=arguments if isinstance(arguments, dict) else {},
                )
            )

        return Reply(text=text, tool_calls=tuple(calls))


class FallbackLLMClient(LLMClient):
    """A second model, asked only when the first cannot answer.

    Only an outage moves the turn to the backup: a timeout, a 5xx, a 429, an
    open circuit. A request the first model refused as malformed would be
    refused by the second as well, so that goes straight up.
    """

    def __init__(self, primary: LLMClient, backup: LLMClient) -> None:
        self.primary = primary
        self.backup = backup

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
    ) -> Reply:
        try:
            return self.primary.complete(messages, tools, tool_choice)
        except LLMUnavailable as exc:
            if exc.code is ErrorCode.INVALID_INPUT or exc.code is ErrorCode.AUTH_FAILURE:
                raise
            log.warning("the main model is unavailable (%s); asking the backup", exc.code)
            return self.backup.complete(messages, tools, tool_choice)
