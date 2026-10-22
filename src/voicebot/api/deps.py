"""What the service holds between requests.

Models load ONCE, at startup, into a single container. Loading per request
would add seconds to every turn and several gigabytes to memory. Every route
reaches them through `get_services`, which is also the seam a test replaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from fastapi import Request

from voicebot.api.sessions import InMemorySessionStore, SessionStore
from voicebot.asr.base import ASREngine, FakeASREngine
from voicebot.config import AppConfig
from voicebot.dialogue.manager import DialogueManager
from voicebot.dialogue.mock_backend import MockBank
from voicebot.nlu.base import FakeNLUEngine, NLUEngine
from voicebot.pipeline.orchestrator import VoicePipeline
from voicebot.tts.base import FakeTTSEngine, TTSEngine
from voicebot.utils.logging import get_logger
from voicebot.utils.seed import set_seed

log = get_logger("api.deps")


@dataclass
class Services:
    """The loaded engines and the session store, built once per process."""

    config: AppConfig
    asr: ASREngine
    nlu: NLUEngine
    tts: TTSEngine
    sessions: SessionStore
    device: str = "cpu"
    turns: int = 0
    action_counts: dict[str, int] = field(default_factory=dict)

    def pipeline(self, bank: MockBank) -> VoicePipeline:
        """A pipeline bound to one conversation's account.

        The engines are shared and stateless, so they are reused. The
        DialogueManager is not: it acts on a bank, and that bank belongs to the
        session. Building it here with the caller's bank is what makes a
        transfer persist across turns without leaking into another session.
        """
        return VoicePipeline(
            asr=self.asr, nlu=self.nlu, tts=self.tts, manager=DialogueManager(bank=bank)
        )

    def record(self, action_type: str) -> None:
        self.turns += 1
        self.action_counts[action_type] = self.action_counts.get(action_type, 0) + 1

    def describe(self) -> dict[str, str]:
        return {
            "asr": type(self.asr).__name__,
            "nlu": type(self.nlu).__name__,
            "tts": type(self.tts).__name__,
        }


def build_services(
    config: AppConfig | None = None, checkpoint: str | Path | None = None
) -> Services:
    """Construct the engines named by the config.

    Falls back to the fakes for any engine set to "fake", which is what makes
    the service startable with no checkpoint and no downloads -- useful for
    testing the HTTP layer on its own.
    """
    config = config or AppConfig()
    set_seed(config.seed)

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

    nlu: NLUEngine
    if config.nlu.engine == "joint":
        from voicebot.nlu.joint_engine import JointNLUEngine

        target = checkpoint or config.nlu.checkpoint
        if not target:
            raise ValueError("nlu.engine is 'joint' but no checkpoint was given")
        nlu = JointNLUEngine.from_checkpoint(target, device=config.nlu.device)
    else:
        nlu = FakeNLUEngine()

    tts: TTSEngine
    if config.tts.engine == "vits":
        from voicebot.tts.vits_engine import VitsTTSEngine

        tts = VitsTTSEngine.load(model_name=config.tts.model_name, device=config.tts.device)
    else:
        tts = FakeTTSEngine()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    services = Services(
        config=config, asr=asr, nlu=nlu, tts=tts, sessions=InMemorySessionStore(), device=device
    )
    log.info("services ready: %s on %s", services.describe(), device)
    return services


def get_services(request: Request) -> Services:
    """FastAPI dependency: hand a route the loaded engines."""
    services = getattr(request.app.state, "services", None)
    if services is None:  # pragma: no cover - only if startup was skipped
        raise RuntimeError("services were never built; check the app's lifespan")
    return services
