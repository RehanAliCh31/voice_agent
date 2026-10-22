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


class NLUConfig(_Strict):
    engine: str = "fake"
    encoder_name: str = "distilbert-base-uncased"
    checkpoint: str = ""
    max_length: int = 64
    oos_threshold: float = 0.5
    device: str = "auto"


# Engines default to "fake" so a bare AppConfig() downloads nothing and
# needs no GPU. Naming a real engine is always a deliberate act.
class TTSConfig(_Strict):
    engine: str = "fake"
    model_name: str = "facebook/mms-tts-eng"
    speaker_index: int = 0
    sample_rate: int = 16000
    device: str = "auto"


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
    """Where the bank lives. `:memory:` for tests, a file for serving."""

    path: str = ":memory:"
    seed_on_create: bool = True


class TrainingConfig(_Strict):
    learning_rate: float = 3e-5
    batch_size: int = 32
    epochs: int = 5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    slot_loss_weight: float = 1.0
    early_stopping_patience: int = 2
    dropout: float = 0.1
    device: str = "auto"  # "auto" | "cuda" | "cpu"


class AppConfig(_Strict):
    seed: int = 42
    data_dir: str = "data"
    asr: ASRConfig = Field(default_factory=ASRConfig)
    nlu: NLUConfig = Field(default_factory=NLUConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)


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
