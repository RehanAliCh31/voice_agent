"""Is every endpoint there, and serving the model the configuration names?

Run at startup and by `check`. Each service is asked for /health and for its
model list; a service that answers but lists a different model is a warning,
not a failure, because the English voice lists itself as
Qwen3-TTS-12Hz-0.6B-Base and still answers to qwen3-tts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

from voicebot.assistant.config import AssistantConfig


@dataclass(frozen=True)
class Check:
    service: str
    url: str
    ok: bool
    latency_ms: int | None
    detail: str
    warning: bool = False


def services(config: AssistantConfig) -> list[tuple[str, str, str]]:
    """(name, endpoint URL, model) for every service the assistant uses."""
    return [
        ("English ASR", config.asr.en.url, config.asr.en.model),
        ("Urdu ASR", config.asr.ur.url, config.asr.ur.model),
        ("English TTS", config.tts.en.url, config.tts.en.model),
        ("Urdu TTS", config.tts.ur.url, config.tts.ur.model),
        ("LLM", config.llm.url, config.llm.model),
    ]


def _base(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


def _model_ids(body: object) -> list[str]:
    """The ids in a /v1/models answer; none, from an answer of any other shape.

    A server that answers its health check but lists its models oddly is
    still up, and is reported as listing none rather than taken for down.
    """
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list):
        return []
    return [m["id"] for m in data if isinstance(m, dict) and isinstance(m.get("id"), str)]


def check_service(name: str, url: str, model: str, client: httpx.Client) -> Check:
    base = _base(url)
    start = time.perf_counter()
    try:
        health = client.get(f"{base}/health")
        latency = round((time.perf_counter() - start) * 1000)
        if health.status_code != 200:
            return Check(name, base, False, latency, f"/health answered {health.status_code}")
        listed = client.get(f"{base}/v1/models")
        models = _model_ids(listed.json()) if listed.status_code == 200 else []
    except (httpx.HTTPError, ValueError) as exc:
        return Check(name, base, False, None, f"unreachable: {type(exc).__name__}: {exc}")
    if model in models:
        return Check(name, base, True, latency, f"serving {model}")
    shown = ", ".join(str(m) for m in models) or "no models listed"
    return Check(
        name, base, True, latency, f"answers, but lists {shown} (configured: {model})", warning=True
    )


def check_all(config: AssistantConfig, client: httpx.Client | None = None) -> list[Check]:
    if client is not None:
        return [check_service(name, url, model, client) for name, url, model in services(config)]
    with httpx.Client(timeout=httpx.Timeout(5.0, connect=3.0)) as own:
        return [check_service(name, url, model, own) for name, url, model in services(config)]


def report(checks: list[Check]) -> str:
    lines = []
    for c in checks:
        mark = "OK  " if c.ok and not c.warning else ("WARN" if c.ok else "DOWN")
        latency = f"{c.latency_ms:>4} ms" if c.latency_ms is not None else "   - ms"
        lines.append(f"  {mark} {c.service:<12} {c.url:<28} {latency}  {c.detail}")
    if any(not c.ok for c in checks):
        lines.append(
            "\n  An endpoint is unreachable. These services are on the office office network:"
            "\n  check this machine is connected to it (or to the VPN)."
        )
    return "\n".join(lines)


def missing_voices(config: AssistantConfig, client: httpx.Client | None = None) -> list[str]:
    """The agent's voices, main and backup, that their server says it does not have.

    A server that lists its voices (OmniVoice: GET /v1/audio/voices) is
    asked; a voice it does not list could only ever fail, in front of a
    caller, so the service refuses to start instead. A server that lists
    none (qwen3-tts) or cannot be reached right now is not held against the
    configuration: the voices' own fallbacks deal with a server that is down.
    """
    wanted = [
        (language, endpoint)
        for language in ("en", "ur")
        for endpoint in (config.tts.voice(language), config.tts.backup(language))  # type: ignore[arg-type]
        if endpoint is not None and endpoint.voice
    ]
    listed: dict[str, set[str] | None] = {}
    own = client is None
    http = client or httpx.Client(timeout=httpx.Timeout(5.0, connect=3.0))
    try:
        for _, endpoint in wanted:
            base = _base(endpoint.url)
            if base not in listed:
                listed[base] = _voices_on(base, http)
    finally:
        if own:
            http.close()
    return [
        f"{language}: {endpoint.voice} is not on {_base(endpoint.url)}"
        for language, endpoint in wanted
        if listed[_base(endpoint.url)] is not None
        and endpoint.voice not in listed[_base(endpoint.url)]  # type: ignore[operator]
    ]


def _voices_on(base: str, client: httpx.Client) -> set[str] | None:
    """The voices a server lists; None when it lists none or cannot be asked."""
    try:
        answer = client.get(f"{base}/v1/audio/voices")
        if answer.status_code != 200:
            return None
        body = answer.json()
    except (httpx.HTTPError, ValueError):
        return None
    names = body.get("voices") if isinstance(body, dict) else None
    if not isinstance(names, list):
        return None
    return {name for name in names if isinstance(name, str)}
