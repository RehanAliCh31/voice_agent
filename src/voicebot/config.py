"""Typed configuration. Every run is driven by one of these objects, and the
resolved object is serialized into the run's result file."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


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
    timeout_s: float = 30.0
    max_retries: int = 2
    temperature: float = 0.4
    max_tokens: int = 200
    # A model that keeps calling tools is stuck; an unbounded loop against a
    # shared endpoint is an outage rather than a bug.
    max_tool_rounds: int = 4
    max_turns: int = 20


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
    pool_size: int = 10


class AppConfig(_Strict):
    seed: int = 42
    asr: ASRConfig = Field(default_factory=ASRConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)


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
    try:
        return AppConfig(**raw)
    except ValidationError as exc:
        raise ValueError(f"Invalid config in {path}:\n{exc}") from exc
