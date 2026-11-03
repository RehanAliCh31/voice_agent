"""What the service holds between requests.

Models and the database are opened ONCE, at startup, into a single container.
Opening them per request would add seconds to every turn. Every route reaches
them through `get_services`, which is also the seam a test replaces.
"""

from __future__ import annotations

import dataclasses
import json
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request

from voicebot import profile
from voicebot.agent.loop import Agent
from voicebot.api.audit import AuditWriter, TurnRecord
from voicebot.api.limits import ConversationLimits
from voicebot.api.payloads import TurnPayload, turn_payload
from voicebot.api.sessions import InMemorySessionStore, Session, SessionStore
from voicebot.asr.base import ASREngine, FakeASREngine
from voicebot.bank import open_bank
from voicebot.bank.base import Bank
from voicebot.config import AppConfig
from voicebot.errors import Cancelled, DatabaseError, InternalError
from voicebot.llm.base import FakeLLMClient, LLMClient, Reply
from voicebot.messages import MessagesError, messages, say
from voicebot.pipeline import spoken
from voicebot.pipeline.orchestrator import TurnResult, VoicePipeline
from voicebot.tts.base import FakeTTSEngine, TTSEngine
from voicebot.utils.logging import get_logger
from voicebot.utils.observability import report
from voicebot.utils.resilience import breakers, configure_breakers
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
    turns: int = 0
    tool_calls: dict[str, int] = field(default_factory=dict)
    audit: AuditWriter | None = None
    limits: ConversationLimits | None = None
    # Set as the service begins to stop: a turn that arrives then is told so,
    # rather than started and cut off.
    draining: bool = False
    _in_flight: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _ready: tuple[float, bool, str] = (0.0, False, "")
    # Makes the live session's voice activity model, one per session (Silero
    # keeps state between frames). Tests put a stand-in here.
    live_vad_factory: Callable[[], Any] | None = None

    def make_live_vad(self) -> Any:
        if self.live_vad_factory is not None:
            return self.live_vad_factory()
        from voicebot.assistant.vad import SileroVad

        return SileroVad(self.config.live.vad_model_path)

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
            turn_budget_s=self.config.resilience.turn_budget_s,
        )
        return VoicePipeline(asr=self.asr, agent=agent, tts=self.tts)

    @property
    def in_flight(self) -> int:
        with self._lock:
            return self._in_flight

    # --- a turn --------------------------------------------------------------

    def take_turn(
        self,
        session_id: str,
        work: Callable[[Session, VoicePipeline], TurnResult],
        record: bool = True,
    ) -> TurnPayload:
        """Run one turn of `session_id`'s conversation, and always answer it.

        One turn per conversation at a time; inside the conversation's limits;
        and whatever fails, a reply the customer can hear. Only two things
        leave here as exceptions: Cancelled, when there is nobody left to
        answer, and an HTTPException a route raised on purpose -- a wrong PIN.
        """
        session = self.sessions.get(session_id)
        language = session.conversation.language
        if self.draining:
            pipeline = self.pipeline(session)
            return self.publish(
                session_id,
                session,
                pipeline.say(say("restarting", language), key="restarting"),
                record=False,
            )
        if not session.lock.acquire(timeout=self.config.session.busy_wait_s):
            # Still answering the last request for this conversation. Not run
            # beside it: two turns on one history, or two Yes presses on one
            # transfer, is how a payment happens twice.
            log.info("a second request arrived while a turn was still running")
            pipeline = self.pipeline(session)
            return self.publish(
                session_id, session, pipeline.say(say("busy", language), key="busy"), record=False
            )
        with self._lock:
            self._in_flight += 1
        if session.speaking is not None:
            # Whatever was left of the last answer is not to be heard now.
            session.speaking.cancel()
            session.speaking = None
        try:
            pipeline = self.pipeline(session)
            limits = self.limits or ConversationLimits(self.config.session)
            if limits.over(session):
                result = limits.close(session, pipeline)
            else:
                try:
                    result = work(session, pipeline)
                except (Cancelled, HTTPException):
                    raise
                except Exception as exc:  # any bug: the caller still hears a sentence
                    # The pipeline has a net of its own; this is the one under it.
                    log.error("the turn failed: %s", type(exc).__name__, exc_info=exc)
                    report(InternalError(f"the turn failed: {type(exc).__name__}", cause=exc))
                    result = pipeline.say(
                        say("internal", session.conversation.language), key="internal"
                    )
                result = limits.settle(session, pipeline, result)
            session.speaking = result.rest
            session.last_result = result
            self.sessions.put(session_id, session)
            return self.publish(session_id, session, result, record=record)
        finally:
            with self._lock:
                self._in_flight -= 1
            session.lock.release()

    def publish(
        self, session_id: str, session: Session, result: TurnResult, record: bool = True
    ) -> TurnPayload:
        """Count it, record it, and convert it into the published shape."""
        if record:
            self.record_turn(session_id, session.conversation.turn_index, result)
        return turn_payload(
            session_id,
            result,
            awaiting_confirmation=session.executor.pending is not None,
            bank=self.bank,
        )

    def record_turn(self, session_id: str, turn_index: int, turn: TurnResult) -> None:
        """Count the turn and queue its record for the audit log."""
        with self._lock:
            self.turns += 1
            for call in turn.agent.tool_calls:
                self.tool_calls[call.name] = self.tool_calls.get(call.name, 0) + 1

        self.record(
            TurnRecord(
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
        )

    def record(self, record: TurnRecord) -> None:
        """Write a turn record -- after the turn, never as part of it."""
        if self.audit is not None:
            self.audit.submit(record)
            return
        try:
            self.bank.log_turn(**dataclasses.asdict(record))
        except Exception as exc:  # the turn is done; a failed record must not undo that
            report(
                DatabaseError("the turn record could not be written", component="audit", cause=exc)
            )

    # --- health --------------------------------------------------------------

    def readiness(self, every_s: float = 3.0) -> tuple[bool, dict[str, str]]:
        """Whether a turn could be served now, and what each dependency says.

        The database is asked at most every `every_s`, so a probe hammering
        /ready does not become load of its own. A provider whose circuit is
        open is reported but does not make the service unready: the bot still
        answers, in a degraded way, and taking it out of rotation would leave
        the customer with no answer at all.
        """
        checks: dict[str, str] = {}
        now = time.monotonic()
        checked_at, database_ok, detail = self._ready
        if now - checked_at >= every_s:
            try:
                self.bank.ping()
                database_ok, detail = True, "ok"
            except Exception as exc:  # whatever it is, the answer is "not ready"
                database_ok, detail = False, f"unreachable ({type(exc).__name__})"
            self._ready = (now, database_ok, detail)
        checks["database"] = detail
        for name, found in breakers().items():
            checks[f"circuit:{name}"] = str(found.state)
        if self.audit is not None:
            checks["audit_queue"] = str(self.audit.pending)
        checks["draining"] = "yes" if self.draining else "no"
        return database_ok and not self.draining, checks

    def warm(self) -> None:
        """Record the bot's own sentences in the background, after startup.

        Callers go first: a recording waits while any turn is running.
        """
        warm = getattr(self.tts, "warm", None)
        if not callable(warm):
            return
        try:
            warm(busy=lambda: self.in_flight > 0)
        except Exception as exc:  # a voice that is down now is asked again on first use
            log.warning("could not record the fallback sentences yet: %s", type(exc).__name__)

    def drain(self, timeout_s: float = 10.0) -> None:
        """Stop taking new turns, and wait for the ones running to finish."""
        self.draining = True
        finish = time.monotonic() + timeout_s
        while self.in_flight and time.monotonic() < finish:
            time.sleep(0.05)
        if self.in_flight:
            log.warning("%d turn(s) still running at shutdown", self.in_flight)

    def close(self) -> None:
        """Release everything, each part on its own: one that fails does not stop the rest."""
        parts: list[tuple[str, Callable[[], None]]] = []
        if self.audit is not None:
            parts.append(("the audit writer", self.audit.close))
        parts += [
            ("the voices' background work", spoken.shutdown),
            ("the recognisers", self.asr.close),
            ("the voices", self.tts.close),
            # A pooled bank holds server connections; hand them back rather
            # than leaving the database to time them out.
            ("the bank", self.bank.close),
        ]
        for what, close in parts:
            try:
                close()
            except Exception as exc:  # shutting down: log it and carry on with the rest
                log.warning("closing %s failed: %s", what, exc)

    def describe(self) -> dict[str, str]:
        return {
            "asr": type(self.asr).__name__,
            "llm": type(self.llm).__name__,
            "tts": type(self.tts).__name__,
        }


def check_startup(config: AppConfig) -> None:
    """Refuse to start, with every reason at once, when the service could not work.

    Checked before anything is opened, so a deployment missing a setting stops
    here, saying which, rather than on the first caller's first turn.
    """
    problems: list[str] = []
    if config.database.engine == "postgres" and not os.environ.get("VOICEBOT_DB_USER"):
        problems.append(
            "database.engine is postgres but VOICEBOT_DB_USER is not set "
            "(set VOICEBOT_DB_USER and VOICEBOT_DB_PASSWORD in .env or the environment)"
        )
    if config.llm.engine == "openai":
        if not config.llm.base_url.startswith(("http://", "https://")):
            problems.append(f"llm.base_url must be an http(s) URL, not {config.llm.base_url!r}")
        if not config.llm.model:
            problems.append("llm.model is empty")
        if bool(config.llm.backup_base_url) != bool(config.llm.backup_model):
            problems.append("a backup model needs both llm.backup_base_url and llm.backup_model")
    for kind, engine in (("asr", config.asr), ("tts", config.tts)):
        if engine.engine == "endpoints" and not Path(engine.endpoints).is_file():
            problems.append(f"{kind}.endpoints names {engine.endpoints}, which does not exist")
    try:
        messages()
    except MessagesError as exc:
        problems.append(str(exc))
    if problems:
        raise RuntimeError("the service cannot start:\n  - " + "\n  - ".join(problems))


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
    configure_breakers(config.resilience.breaker_failures, config.resilience.breaker_cooldown_s)

    # Config is strict about unknown keys; it has to be strict about unknown
    # values too. A single typo would otherwise start cleanly and answer every
    # question with "the model isn't configured yet".
    _check_engine("asr", config.asr.engine, {"endpoints", "fake"})
    _check_engine("llm", config.llm.engine, {"openai", "fake"})
    _check_engine("tts", config.tts.engine, {"endpoints", "fake"})
    _check_engine("database", config.database.engine, {"sqlite", "postgres"})

    # Who the bot is -- its gender, its name, its voices -- comes from the
    # voices' file, the one place it is set. Without the office voices the
    # default profile stands.
    if "endpoints" in (config.tts.engine, config.asr.engine):
        from voicebot.assistant.config import load_assistant_config

        endpoints = (
            config.tts.endpoints if config.tts.engine == "endpoints" else config.asr.endpoints
        )
        voices = load_assistant_config(endpoints)
        profile.use(voices.profile())
        if config.tts.engine == "endpoints":
            from voicebot.assistant.health import missing_voices

            # A voice its server does not have could only fail in front of a
            # caller, and its backup is not the voice chosen. Refused here,
            # at startup, by name.
            missing = missing_voices(voices)
            if missing:
                raise RuntimeError("the service cannot start:\n  - " + "\n  - ".join(missing))
    log.info("the agent is %s, %s", profile.current().name("en"), profile.current().gender)

    asr: ASREngine
    if config.asr.engine == "endpoints":
        from voicebot.asr.endpoints import EndpointsASREngine
        from voicebot.assistant.config import load_assistant_config

        asr = EndpointsASREngine.from_config(load_assistant_config(config.asr.endpoints))
    else:
        asr = FakeASREngine()

    llm: LLMClient
    if config.llm.engine == "openai":
        from voicebot.llm.openai_client import FallbackLLMClient, OpenAIClient

        if not config.llm.base_url or not config.llm.model:
            raise ValueError("llm.engine is 'openai' but base_url or model is missing")

        def client(base_url: str, model: str, name: str) -> OpenAIClient:
            return OpenAIClient(
                base_url=base_url,
                model=model,
                api_key=os.environ.get(config.llm.api_key_env) or None,
                timeout_s=config.llm.timeout_s,
                max_retries=config.llm.max_retries,
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens,
                backoff_s=config.resilience.retry_base_s,
                backoff_cap_s=config.resilience.retry_cap_s,
                name=name,
                disable_thinking=config.llm.disable_thinking,
            )

        llm = client(config.llm.base_url, config.llm.model, "llm")
        if config.llm.backup_base_url and config.llm.backup_model:
            log.info("a backup model is configured: %s", config.llm.backup_model)
            llm = FallbackLLMClient(
                llm, client(config.llm.backup_base_url, config.llm.backup_model, "llm-backup")
            )
    else:
        # A scripted client that says one neutral thing, so the service starts
        # and the page loads with no endpoint configured.
        llm = FakeLLMClient([Reply(text="The model isn't configured yet.")] * 10_000)

    tts: TTSEngine
    if config.tts.engine == "endpoints":
        from voicebot.assistant.config import load_assistant_config
        from voicebot.tts.endpoints import EndpointsTTSEngine

        tts = EndpointsTTSEngine.from_config(load_assistant_config(config.tts.endpoints))
    else:
        tts = FakeTTSEngine()

    bank = open_bank(config.database)

    services = Services(
        config=config,
        asr=asr,
        llm=llm,
        tts=tts,
        bank=bank,
        sessions=InMemorySessionStore(
            bank=bank,
            ttl_seconds=config.session.ttl_s,
            max_sessions=config.session.max_sessions,
        ),
        limits=ConversationLimits(config.session),
    )
    # Told how to find the bank rather than given one: records go wherever
    # services.bank points, the one place every route reads it from.
    services.audit = AuditWriter(
        lambda: services.bank,
        config.audit.dead_letter,
        retries=config.audit.retries,
        queue_size=config.audit.queue_size,
    )
    log.info("services ready: %s", services.describe())
    return services


def get_services(request: Request) -> Services:
    """FastAPI dependency: hand a route the loaded engines."""
    services = getattr(request.app.state, "services", None)
    if services is None:  # pragma: no cover - only if startup was skipped
        raise RuntimeError("services were never built; check the app's lifespan")
    return services
