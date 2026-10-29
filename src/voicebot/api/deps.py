"""What the service holds between requests.

Models and the database are opened ONCE, at startup, into a single container.
Opening them per request would add seconds to every turn. Every route reaches
them through `get_services`, which is also the seam a test replaces.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from fastapi import Request

from voicebot.agent.loop import Agent
from voicebot.api.sessions import InMemorySessionStore, Session, SessionStore
from voicebot.asr.base import ASREngine, FakeASREngine
from voicebot.bank import open_bank
from voicebot.bank.base import Bank
from voicebot.config import AppConfig
from voicebot.llm.base import FakeLLMClient, LLMClient, Reply
from voicebot.pipeline.orchestrator import TurnResult, VoicePipeline
from voicebot.tts.base import FakeTTSEngine, TTSEngine
from voicebot.utils.logging import get_logger
from voicebot.utils.seed import set_seed

log = get_logger("api.deps")


@dataclass
class Services:
    """The loaded engines, the bank and the session store, built once."""

    config: AppConfig
    asr: ASREngine
    llm: LLMClient
    tts: TTSEngine
    bank: Bank
    sessions: SessionStore
    device: str = "cpu"
    turns: int = 0
    tool_calls: dict[str, int] = field(default_factory=dict)

    def pipeline(self, session: Session) -> VoicePipeline:
        """A pipeline bound to one caller's conversation.

        The engines and the bank are shared; the executor is not, because it
        holds that caller's pending confirmation. Two tabs mid-confirmation
        must not be able to confirm each other's transfer.
        """
        agent = Agent(
            client=self.llm,
            executor=session.executor,
            max_tool_rounds=self.config.llm.max_tool_rounds,
            max_turns=self.config.llm.max_turns,
        )
        return VoicePipeline(asr=self.asr, agent=agent, tts=self.tts)

    @property
    def speaks_client_side(self) -> bool:
        return bool(getattr(self.tts, "speaks_client_side", False))

    def record_turn(self, session_id: str, turn_index: int, turn: TurnResult) -> None:
        """Count the turn and write it to the audit log."""
        self.turns += 1
        for call in turn.agent.tool_calls:
            self.tool_calls[call.name] = self.tool_calls.get(call.name, 0) + 1

        self.bank.log_turn(
            session_id=session_id,
            turn_index=turn_index,
            user_text=turn.text,
            bot_text=turn.agent.text,
            # NULL, not "[]": json.dumps([]) is truthy, so `or None` never
            # fired and the schema's "nullable" was a fiction.
            tool_calls=(
                json.dumps([call.name for call in turn.agent.tool_calls])
                if turn.agent.tool_calls
                else None
            ),
            guard_hit=turn.agent.guard_hit,
            suppressed=turn.agent.suppressed,
            latency_ms=turn.latency.total_ms,
        )

    def describe(self) -> dict[str, str]:
        return {
            "asr": type(self.asr).__name__,
            "llm": type(self.llm).__name__,
            "tts": type(self.tts).__name__,
        }


def _check_engine(kind: str, name: str, known: set[str]) -> None:
    """Refuse an engine name nothing implements, rather than falling back."""
    if name not in known:
        raise ValueError(f"unknown {kind} engine {name!r}. Known: {', '.join(sorted(known))}.")


def build_services(config: AppConfig | None = None) -> Services:
    """Construct the engines, the bank and the session store named by the config.

    Falls back to fakes for any engine set to "fake", which is what makes the
    service startable with no endpoint and no downloads.
    """
    config = config or AppConfig()
    set_seed(config.seed)

    # Config is strict about unknown keys; it has to be strict about unknown
    # values too. A single typo would otherwise start cleanly and answer every
    # question with "the model isn't configured yet".
    _check_engine("asr", config.asr.engine, {"whisper", "fake"})
    _check_engine("llm", config.llm.engine, {"openai", "fake"})
    _check_engine("tts", config.tts.engine, {"vits", "browser", "fake"})
    _check_engine("database", config.database.engine, {"sqlite", "postgres"})

    asr: ASREngine
    if config.asr.engine == "whisper":
        from voicebot.asr.whisper_engine import WhisperASREngine

        asr = WhisperASREngine.load(
            model_name=config.asr.model_name,
            device=config.asr.device,
            language=config.asr.language,
        )
    else:
        asr = FakeASREngine()

    llm: LLMClient
    if config.llm.engine == "openai":
        from voicebot.llm.openai_client import OpenAIClient

        if not config.llm.base_url or not config.llm.model:
            raise ValueError("llm.engine is 'openai' but base_url or model is missing")
        llm = OpenAIClient(
            base_url=config.llm.base_url,
            model=config.llm.model,
            api_key=os.environ.get(config.llm.api_key_env) or None,
            timeout_s=config.llm.timeout_s,
            max_retries=config.llm.max_retries,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
        )
    else:
        # A scripted client that says one neutral thing, so the service starts
        # and the page loads with no endpoint configured.
        llm = FakeLLMClient([Reply(text="The model isn't configured yet.")] * 10_000)

    tts: TTSEngine
    if config.tts.engine == "vits":
        from voicebot.tts.vits_engine import VitsTTSEngine

        tts = VitsTTSEngine.load(model_name=config.tts.model_name, device=config.tts.device)
    elif config.tts.engine == "browser":
        from voicebot.tts.browser import BrowserTTSEngine

        tts = BrowserTTSEngine()
    else:
        tts = FakeTTSEngine()

    bank = open_bank(config.database)

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    services = Services(
        config=config,
        asr=asr,
        llm=llm,
        tts=tts,
        bank=bank,
        sessions=InMemorySessionStore(bank=bank),
        device=device,
    )
    log.info("services ready: %s on %s", services.describe(), device)
    return services


def get_services(request: Request) -> Services:
    """FastAPI dependency: hand a route the loaded engines."""
    services = getattr(request.app.state, "services", None)
    if services is None:  # pragma: no cover - only if startup was skipped
        raise RuntimeError("services were never built; check the app's lifespan")
    return services
