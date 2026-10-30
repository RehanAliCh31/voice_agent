"""Typed configuration. Every run is driven by one of these objects, and the
resolved object is serialized into the run's result file."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class _Strict(BaseModel):
    """Base that rejects unknown keys, so a typo in YAML fails loudly."""

    model_config = ConfigDict(extra="forbid")


# Each engine names its own device: running ASR on the GPU and TTS on the
# CPU is a real tuning choice when VRAM is tight.
class ASRConfig(_Strict):
    engine: str = "fake"
    model_name: str = "openai/whisper-small"
    device: str = "auto"
    language: str = "en"
    # `engine: endpoints` uses the office recognisers, English and Urdu, as
    # listed in this file -- the one place their addresses are kept.
    endpoints: str = "configs/assistant.yaml"


# Engines default to "fake" so a bare AppConfig() downloads nothing and
# needs no GPU. Naming a real engine is always a deliberate act.
class TTSConfig(_Strict):
    engine: str = "fake"
    model_name: str = "facebook/mms-tts-eng"
    speaker_index: int = 0
    sample_rate: int = 16000
    device: str = "auto"
    # `engine: endpoints` uses the office voices, English and Urdu, as listed
    # in this file.
    endpoints: str = "configs/assistant.yaml"


class LLMConfig(_Strict):
    """The model that does the understanding and the talking.

    `engine: fake` selects the scripted client, so a bare AppConfig still
    starts the service with no network and no downloads. The key is named
    rather than stored, so a config file is safe to commit.
    """

    engine: str = "fake"
    base_url: str = ""
    model: str = ""
    api_key_env: str = "VOICEBOT_LLM_KEY"
    # One request. The caller is listening to silence for all of it: a long
    # reply measured 9.4 s, so 12 s fails a dead server fast without cutting
    # a slow answer off. The turn's own budget (resilience.turn_budget_s)
    # bounds every round together.
    timeout_s: float = Field(default=12.0, gt=0)
    max_retries: int = Field(default=1, ge=0, le=3)
    temperature: float = 0.4
    max_tokens: int = Field(default=200, gt=0)
    # Asks the server not to think before answering (chat_template_kwargs).
    # Thinking is never shown or spoken, but it is waited for and it counts
    # against max_tokens: a reply that thought first was cut off part way.
    disable_thinking: bool = True
    # A model that keeps calling tools is stuck; an unbounded loop against a
    # shared endpoint is an outage rather than a bug.
    max_tool_rounds: int = Field(default=4, ge=1)
    max_turns: int = Field(default=20, ge=1)
    # A second model to ask when the first is down. Empty: there is none,
    # and the customer hears the "unavailable" sentence instead.
    backup_base_url: str = ""
    backup_model: str = ""


class DatabaseConfig(_Strict):
    """Where the bank lives.

    `sqlite` keeps it in a file, or in memory for tests and the offline
    path. `postgres` puts it on a server, reached with the VOICEBOT_DB_*
    environment variables -- the connection is not configured here, because
    a config file is committed and a password must never be.
    """

    engine: str = "sqlite"
    path: str = ":memory:"
    seed_on_create: bool = True
    # The most connections the PostgreSQL bank holds open at once.
    pool_size: int = Field(default=10, ge=1)
    # How long a request waits for a free connection before it is refused.
    pool_timeout_s: float = Field(default=5.0, gt=0)
    # The longest any one statement may run, and wait for a lock, on the
    # server. Without them a hung query holds a caller's turn for ever.
    statement_timeout_ms: int = Field(default=5000, gt=0)
    lock_timeout_ms: int = Field(default=3000, gt=0)


class ResilienceConfig(_Strict):
    """How provider calls are retried and cut off. See voicebot.utils.resilience."""

    # Every model round of one turn together. Past it the turn ends with a
    # sentence rather than another round.
    turn_budget_s: float = Field(default=20.0, gt=0)
    retry_base_s: float = Field(default=0.2, ge=0)
    retry_cap_s: float = Field(default=1.0, ge=0)
    # A provider is skipped after this many failures in a row, for this long.
    breaker_failures: int = Field(default=3, ge=1)
    breaker_cooldown_s: float = Field(default=20.0, gt=0)


class SessionConfig(_Strict):
    """How long a conversation may run, and when it is handed to a person."""

    ttl_s: float = Field(default=1800.0, gt=0)
    max_sessions: int = Field(default=1000, ge=1)
    # The longest conversation, in turns and in minutes. It is warned about
    # `warn_turns` turns before the end, then closed politely.
    max_turns: int = Field(default=60, ge=2)
    max_minutes: float = Field(default=30.0, gt=0)
    warn_turns: int = Field(default=3, ge=0)
    # Turns in a row that end in a fallback -- not heard, not answered, a
    # lookup failed -- before a colleague is asked to take over.
    failure_limit: int = Field(default=3, ge=1)
    # How long a second request for a conversation waits for the first.
    busy_wait_s: float = Field(default=2.0, ge=0)


class LiveConfig(_Strict):
    """The hands-free session: how speech is found, ended and interrupted.

    Probabilities are Silero VAD's, per 32 ms frame. Durations are of audio,
    so they hold whatever the network does. See HANDS_FREE.md for tuning.
    """

    vad_model_path: str = "models/silero/silero_vad.onnx"
    # Speech starts above `threshold` and carries on until it falls below
    # `neg_threshold`, so a soft syllable mid-word does not end it.
    threshold: float = Field(default=0.5, gt=0, lt=1)
    neg_threshold: float = Field(default=0.35, gt=0, lt=1)
    # Shorter than this is a cough, a click or a chair: not speech.
    min_speech_ms: int = Field(default=200, ge=32)
    # Kept from before speech was confirmed, so its first syllable is heard.
    pre_roll_ms: int = Field(default=300, ge=0)
    # The silence that ends a turn. Shorter answers sooner and cuts more
    # people off mid-thought; speech that resumes before the reply is heard
    # joins the same turn whatever this is (see HANDS_FREE.md).
    end_silence_ms: int = Field(default=500, ge=200, le=2000)
    max_utterance_s: float = Field(default=20.0, gt=1)
    # While the bot is speaking, what may interrupt it is surer, so its own
    # voice coming back through the microphone does not.
    barge_in_threshold: float = Field(default=0.7, gt=0, lt=1)
    barge_in_min_speech_ms: int = Field(default=192, ge=32)
    # An interruption no longer than this whose words are only "mm-hmm",
    # "ok", "جی" ... was not one: the reply carries on where it stopped.
    backchannel_max_ms: int = Field(default=800, ge=0)
    backchannels: list[str] = Field(
        default_factory=lambda: [
            "mm",
            "mhm",
            "mm hmm",
            "uh huh",
            "hmm",
            "ok",
            "okay",
            "yeah",
            "yes",
            "right",
            "sure",
            "جی",
            "جی ہاں",
            "ہاں",
            "اچھا",
            "ٹھیک",
            "ٹھیک ہے",
            "ہوں",
            "ہم",
        ]
    )
    # Said nothing for this long: asked whether they are still there; as long
    # again after that, and the session is ended politely.
    idle_prompt_s: float = Field(default=20.0, gt=1)
    idle_end_s: float = Field(default=20.0, gt=1)
    # The longest a live session runs, whatever happens in it.
    max_session_minutes: float = Field(default=30.0, gt=0)
    # If the page never says the reply finished playing, it is taken to have
    # finished this long after its audio would have.
    playback_margin_s: float = Field(default=3.0, ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> LiveConfig:
        if not self.neg_threshold <= self.threshold <= self.barge_in_threshold:
            raise ValueError("live: need neg_threshold <= threshold <= barge_in_threshold")
        return self


class AuditConfig(_Strict):
    """The record of every turn, written after the turn rather than during it."""

    retries: int = Field(default=3, ge=0)
    queue_size: int = Field(default=1000, ge=1)
    # Where a turn the database would not take is kept, one JSON line each,
    # so a failed save is never silently lost.
    dead_letter: str = "data/dead_letter/turns.jsonl"


class AppConfig(_Strict):
    seed: int = 42
    asr: ASRConfig = Field(default_factory=ASRConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    resilience: ResilienceConfig = Field(default_factory=ResilienceConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    live: LiveConfig = Field(default_factory=LiveConfig)


ENV_PREFIX = "VOICEBOT__"


def env_overrides(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Settings from the environment: VOICEBOT__LLM__TIMEOUT_S=8 sets llm.timeout_s.

    Double underscores separate the levels, so a deployment can tighten a
    timeout without editing a committed file. The value is checked like any
    other.
    """
    environ = os.environ if environ is None else environ
    found: dict[str, Any] = {}
    for name, value in environ.items():
        if not name.startswith(ENV_PREFIX):
            continue
        path = [part.lower() for part in name[len(ENV_PREFIX) :].split("__") if part]
        if not path:
            continue
        node = found
        for part in path[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"{name} sets a value inside one that is not a section")
        node[path[-1]] = value
    return found


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def build_config(raw: dict[str, Any], source: str = "defaults") -> AppConfig:
    """`raw` with the environment's overrides applied, checked."""
    try:
        return AppConfig(**_merge(raw, env_overrides()))
    except ValidationError as exc:
        raise ValueError(f"Invalid config in {source}:\n{exc}") from exc


def load_config(path: str | Path) -> AppConfig:
    """Load a YAML file into an AppConfig, merging over defaults."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"Config file {path} must contain a YAML mapping, got {type(raw).__name__}"
        )
    return build_config(raw, str(path))
