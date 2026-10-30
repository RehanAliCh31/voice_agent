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

from voicebot.profile import AgentProfile, Gender

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
    # A recording to clone the voice from, sent inline with every request
    # (OmniVoice's ref_audio and ref_text), so nothing has to be uploaded to
    # the server: a path to a WAV, and exactly what it says.
    reference: str | None = None
    reference_text: str | None = None
    # Whose voice this is. Checked against the agent's gender at startup, so
    # a female agent can never be given a male voice, or a male backup.
    gender: Gender | None = None
    # Shown in a voice's listing and the samples; not sent to the server.
    label: str = ""

    def identity(self) -> str:
        """What makes two voices the same voice: a recording of one is one of the other."""
        import hashlib

        extra = ",".join(f"{k}={self.extra[k]!r}" for k in sorted(self.extra) if k != "ref_audio")
        ref = self.extra.get("ref_audio")
        if isinstance(ref, str):
            extra += f",ref_audio={hashlib.sha1(ref.encode()).hexdigest()[:12]}"
        if self.reference:
            try:
                digest = hashlib.sha1(Path(self.reference).read_bytes()).hexdigest()[:12]
            except OSError:
                digest = self.reference
            extra += f",reference={digest}"
        return f"{self.url}|{self.model}|{self.voice or ''}|{extra}"


class TtsConfig(_Strict):
    # Filled from `agent.voices` for the agent's gender when the file has an
    # `agent:` section; written here only by a file without one.
    en: TtsEndpoint
    ur: TtsEndpoint
    # Tried when the voice above fails, before a kept recording. Same gender.
    backup_en: TtsEndpoint | None = None
    backup_ur: TtsEndpoint | None = None
    timeout_s: float = 20.0
    retries: int = 1
    # Both voices start with up to 0.4 s of silence. Cut, the first word
    # arrives that much sooner.
    trim_silence: bool = True

    def voice(self, language: Language) -> TtsEndpoint:
        return self.ur if language == "ur" else self.en

    def backup(self, language: Language) -> TtsEndpoint | None:
        return self.backup_ur if language == "ur" else self.backup_en


# --- who the bot is -------------------------------------------------------------


class Voices(_Strict):
    en: TtsEndpoint
    ur: TtsEndpoint


class BackupVoices(_Strict):
    en: TtsEndpoint | None = None
    ur: TtsEndpoint | None = None


class AgentConfig(_Strict):
    """The bot's profile: its gender, its name, and its voice in each language.

    `gender` is the one switch. The voices, the name, every fixed sentence
    and how the model is told to speak of itself all follow it. Each voice
    says whose it is, and a voice of the other gender -- main or backup --
    stops the service at startup rather than reaching a caller.
    """

    gender: Gender
    names: dict[Gender, Bilingual]
    voices: dict[Gender, Voices]
    backup_voices: dict[Gender, BackupVoices] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _consistent(self) -> AgentConfig:
        problems = []
        if self.gender not in self.voices:
            problems.append(f"agent.voices has no {self.gender} voices for a {self.gender} agent")
        if self.gender not in self.names:
            problems.append(f"agent.names has no name for a {self.gender} agent")
        for gender, voices in self.voices.items():
            for language in LANGUAGES:
                voice: TtsEndpoint = getattr(voices, language)
                if voice.reference and not Path(voice.reference).is_file():
                    problems.append(
                        f"agent.voices.{gender}.{language}.reference: no file at {voice.reference}"
                    )
                if voice.reference and not voice.reference_text:
                    problems.append(
                        f"agent.voices.{gender}.{language}: a reference needs its reference_text"
                    )
                if voice.gender != gender:
                    problems.append(
                        f"agent.voices.{gender}.{language} is "
                        f"{voice.gender or 'of no stated gender'}, not {gender}"
                    )
        for gender, backups in self.backup_voices.items():
            for language in LANGUAGES:
                backup: TtsEndpoint | None = getattr(backups, language)
                if backup is not None and backup.gender != gender:
                    problems.append(
                        f"agent.backup_voices.{gender}.{language} is "
                        f"{backup.gender or 'of no stated gender'}, not {gender}"
                    )
        if problems:
            raise ValueError("; ".join(problems))
        return self

    def profile(self) -> AgentProfile:
        names = self.names[self.gender]
        return AgentProfile(gender=self.gender, names={"en": names.en, "ur": names.ur})


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
    # Who the bot is. Without it the default profile stands -- female -- and
    # tts.en and tts.ur must be given directly.
    agent: AgentConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def _voices_from_the_agent(cls, raw: Any) -> Any:
        """The agent's voices, for its gender, become tts.en and tts.ur.

        One place says which voice the bot has, so tts.en and tts.ur may not
        also be written by hand: two places would sooner or later disagree.
        """
        if not isinstance(raw, dict) or not isinstance(raw.get("agent"), dict):
            return raw
        agent, tts = raw["agent"], dict(raw.get("tts") or {})
        written = [key for key in ("en", "ur", "backup_en", "backup_ur") if key in tts]
        if written:
            raise ValueError(
                f"tts.{', tts.'.join(written)} cannot be set when there is an agent: "
                "its voices are set in agent.voices and agent.backup_voices"
            )
        gender = agent.get("gender")
        voices = (agent.get("voices") or {}).get(gender) or {}
        backups = (agent.get("backup_voices") or {}).get(gender) or {}
        for language in LANGUAGES:
            if language in voices:
                tts[language] = voices[language]
            if backups.get(language) is not None:
                tts[f"backup_{language}"] = backups[language]
        return {**raw, "tts": tts}

    def profile(self) -> AgentProfile:
        """The bot's profile: from `agent:`, or the default without one."""
        return self.agent.profile() if self.agent else AgentProfile()


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
