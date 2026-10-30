"""The assistant's configuration: every endpoint, model, voice and threshold.

No host, port, model name or voice appears in code. They live in
configs/assistant.yaml, which holds no secrets and is committed; a different
file can be named with VOICEBOT_ASSISTANT_CONFIG. The one secret, the LLM's
key if it ever needs one, is read from the environment variable the file
names.

Relative paths in the file are relative to the directory the assistant is
started from, the project root, as elsewhere in this project.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

Language = Literal["en", "ur"]
LANGUAGES: tuple[Language, ...] = ("en", "ur")

DEFAULT_CONFIG = Path("configs/assistant.yaml")
CONFIG_ENV = "VOICEBOT_ASSISTANT_CONFIG"


class _Strict(BaseModel):
    """Rejects unknown keys, so a typo in the YAML fails loudly."""

    model_config = ConfigDict(extra="forbid")


class Bilingual(_Strict):
    """One value per language."""

    en: str
    ur: str

    def __getitem__(self, language: Language) -> str:
        return str(getattr(self, language))


class BilingualList(_Strict):
    en: list[str] = Field(default_factory=list)
    ur: list[str] = Field(default_factory=list)

    def __getitem__(self, language: Language) -> list[str]:
        return list(getattr(self, language))


# --- the services -------------------------------------------------------------


class AsrEndpoint(_Strict):
    url: str
    model: str
    # verbose_json carries the language Whisper heard and how sure it was;
    # qwen3-asr refuses it, so each endpoint names its own.
    response_format: str = "json"


class AsrConfig(_Strict):
    en: AsrEndpoint
    ur: AsrEndpoint
    timeout_s: float = 10.0
    retries: int = 1
    # Whisper is sure of real speech: -0.08 to -0.22 measured. Noise came in
    # at -1.03, digital silence at -0.44.
    min_avg_logprob: float = -0.8
    # What the recognisers say about silence, which VAD should keep from them
    # but which one stray click must not turn into a question.
    phantom_phrases: BilingualList = Field(default_factory=BilingualList)
    phantom_max_speech_ms: int = 1500


class TtsEndpoint(_Strict):
    url: str
    model: str
    voice: str | None = None
    # Anything else the server takes, passed through as is (a seed, say).
    extra: dict[str, Any] = Field(default_factory=dict)
    # Nothing is heard until the first piece of a reply is synthesised, so it
    # is cut short: at its first comma, or after this many words (0: wait for
    # the whole sentence). `chunk_growth` lets the next few pieces each be
    # that many times longer, so each is ready before the last one ends; 0
    # goes straight to whole sentences. Per voice, because the voices differ:
    # English synthesises at 0.6x real time, Urdu with ~1.1 s per request.
    first_chunk_words: int = 8
    chunk_growth: float = 0.0


class TtsConfig(_Strict):
    en: TtsEndpoint
    ur: TtsEndpoint
    timeout_s: float = 20.0
    retries: int = 1
    # Both voices start with up to 0.4 s of silence. Cut, the first word
    # arrives that much sooner.
    trim_silence: bool = True


class LlmConfig(_Strict):
    url: str
    model: str
    api_key_env: str = "VOICEBOT_LLM_KEY"
    max_tokens: int = 150
    temperature: float = 0.4
    timeout_s: float = 20.0
    # The whole reply, retries included. `timeout_s` bounds each wait for the
    # next piece; a server that keeps sending a word every few seconds would
    # pass that for ever, so the reply as a whole is bounded too.
    total_timeout_s: float = Field(default=30.0, gt=0)
    retries: int = 1
    stream: bool = True
    # Asked for explicitly, so a change of default on the server cannot
    # start sending reasoning to be read aloud.
    disable_thinking: bool = True
    history_turns: int = 6
    system_prompts: Bilingual


# --- the microphone and the models ----------------------------------------------


class AudioConfig(_Strict):
    sample_rate: Literal[16000] = 16000
    # A name (any part of it) or an index from `devices`. Empty: the default.
    input_device: str | int | None = None
    output_device: str | int | None = None
    # How many frames the microphone may run ahead of the loop before the
    # oldest are dropped. 64 frames is two seconds.
    queue_frames: int = 64


class WakeWordConfig(_Strict):
    phrase: str = "Hey Jarvis"
    model: str = "hey_jarvis_v0.1"
    models_dir: str = "models/openwakeword"
    threshold: float = 0.5
    # After firing, it ignores the phrase for this long, so the tail of one
    # "Hey Jarvis" cannot wake it twice.
    cooldown_ms: int = 1500


class VadConfig(_Strict):
    model_path: str = "models/silero/silero_vad.onnx"
    threshold: float = 0.5
    # Speech carries on until the probability falls below this, so a soft
    # syllable mid-sentence is not taken for the end of it.
    neg_threshold: float = 0.35
    pre_roll_ms: int = 300
    min_speech_ms: int = 250
    end_silence_ms: int = 800
    max_utterance_s: float = 20.0
    no_speech_timeout_s: float = 6.0

    @model_validator(mode="after")
    def _ordered(self) -> VadConfig:
        if not 0 < self.neg_threshold <= self.threshold < 1:
            raise ValueError("vad: need 0 < neg_threshold <= threshold < 1")
        return self


# --- the conversation ---------------------------------------------------------


class Phrases(_Strict):
    """What the assistant says when it has nothing of its own to say."""

    fallback: Bilingual
    goodbye: Bilingual


class Commands(_Strict):
    """Words that end the conversation, matched only when said on their own."""

    stop: BilingualList
    goodbye: BilingualList


class SessionConfig(_Strict):
    language: Literal["auto", "en", "ur"] = "auto"
    follow_up_s: float = 8.0
    max_session_s: float = 300.0
    earcon: bool = True
    # speakers: the microphone is not listened to while the assistant talks,
    # so its own voice cannot interrupt it; saying the wake word still does.
    # headphones: talking over it stops it.
    echo_mode: Literal["speakers", "headphones"] = "speakers"
    echo_tail_ms: int = 300
    barge_in_min_speech_ms: int = 300
    # Words after the wake phrase are kept, but the phrase itself is not.
    post_wake_ignore_ms: int = 0
    phrases: Phrases
    commands: Commands


class DebugConfig(_Strict):
    # Off unless asked for: microphone audio is not written to disk.
    save_audio: bool = False
    audio_dir: str = "debug_audio"


class AssistantConfig(_Strict):
    asr: AsrConfig
    tts: TtsConfig
    llm: LlmConfig
    audio: AudioConfig = Field(default_factory=AudioConfig)
    wake_word: WakeWordConfig = Field(default_factory=WakeWordConfig)
    vad: VadConfig = Field(default_factory=VadConfig)
    session: SessionConfig
    debug: DebugConfig = Field(default_factory=DebugConfig)


def config_path(path: str | Path | None = None) -> Path:
    """The file to read: the one named, else VOICEBOT_ASSISTANT_CONFIG, else the default."""
    return Path(path or os.environ.get(CONFIG_ENV) or DEFAULT_CONFIG)


def load_assistant_config(path: str | Path | None = None) -> AssistantConfig:
    file = config_path(path)
    if not file.is_file():
        raise FileNotFoundError(f"assistant config not found: {file}")
    raw = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{file} must contain a YAML mapping")
    try:
        return AssistantConfig(**raw)
    except ValidationError as exc:
        raise ValueError(f"invalid assistant config in {file}:\n{exc}") from exc
