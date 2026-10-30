"""The HTTP surface.

A whole turn, the two confirmation buttons, a reset, and the operational
endpoints, and a read of the ledger for the page's first paint. The per-stage
debugging routes from v1 are gone with the stages they exposed; what a turn
reports about its own work comes back with the turn itself.
"""

from __future__ import annotations

import asyncio
import base64
import contextvars
import json
import secrets
from collections.abc import Callable
from datetime import date

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import PlainTextResponse

from voicebot import __version__
from voicebot.api.audit import TurnRecord
from voicebot.api.deps import Services, get_services
from voicebot.api.payloads import (
    ActionPayload,
    ActionRequest,
    AudioPayload,
    CardDetailsPayload,
    CardsPayload,
    ConfirmRequest,
    HealthPayload,
    LedgerPayload,
    MonthlyPayload,
    PinChangeRequest,
    PinRequest,
    ReadyPayload,
    SessionRequest,
    SpeakRequest,
    StatementPayload,
    TextRequest,
    TranscriptPayload,
    TurnPayload,
    card_details_payload,
    cards_payload,
    ledger_payload,
    monthly_payload,
    statement_payload,
)
from voicebot.api.sessions import Session, SessionStore
from voicebot.asr.base import ASRUnavailable
from voicebot.bank.base import PIN_LENGTH, BankError, Card
from voicebot.errors import Cancelled, call_context
from voicebot.llm.base import ToolCall
from voicebot.messages import messages, say
from voicebot.pipeline.audio_io import decode_wav, encode_wav
from voicebot.pipeline.orchestrator import TurnResult, VoicePipeline
from voicebot.schemas import AudioBuffer
from voicebot.utils.logging import get_logger
from voicebot.utils.observability import count, metrics_lines
from voicebot.utils.resilience import CancelToken, cancel_context

log = get_logger("api.routes")

router = APIRouter()

MAX_AUDIO_BYTES = 10 * 1024 * 1024  # ~5 minutes of 16 kHz mono


def _read_audio(data: bytes) -> AudioBuffer:
    """Decode an uploaded WAV, refusing anything unusable with a clear reason."""
    if not data:
        raise HTTPException(status_code=400, detail="No audio was uploaded.")
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Audio is {len(data) // 1024} kB; the limit is {MAX_AUDIO_BYTES // 1024} kB.",
        )
    try:
        return decode_wav(data)
    except Exception as exc:  # any decoder failure is the same refusal
        # The decoder's own words ("file does not start with RIFF id") are
        # for the log, not for the page.
        log.warning("an upload could not be decoded: %s", type(exc).__name__)
        raise HTTPException(
            status_code=400, detail="That recording could not be read. Please try again."
        ) from exc


def _session_id(store: SessionStore, given: str | None) -> str:
    return given or store.new_id()


# How often a running turn looks to see whether its caller is still there.
DISCONNECT_POLL_S = 0.25


async def _turn(
    request: Request,
    services: Services,
    session_id: str,
    work: Callable[[Session, VoicePipeline], TurnResult],
) -> TurnPayload | Response:
    """Run one turn on a worker thread, and stop it if the caller leaves.

    A turn is seconds of blocking work -- recognisers, the model, the voice.
    Run on the event loop, as converse once was, it froze the whole server
    for every turn: every other caller, and /health, waited behind it.

    While it runs, the request is watched. A caller who closes the page or
    starts over mid-turn sets the turn's cancel token, and the turn stops at
    its next safe point -- before the model, between rounds, before the
    voice -- rather than paying for work nobody will hear.
    """
    token = CancelToken()

    def run() -> TurnPayload:
        with call_context(session_id), cancel_context(token):
            return services.take_turn(session_id, work)

    job = asyncio.ensure_future(run_in_threadpool(contextvars.copy_context().run, run))
    while not job.done():
        done, _ = await asyncio.wait({job}, timeout=DISCONNECT_POLL_S)
        if not done and await request.is_disconnected():
            token.cancel("the caller left")
    try:
        return await job
    except Cancelled as exc:
        with call_context(session_id):
            log.info("turn abandoned: %s", exc)
        count("turns_abandoned")
        # Nobody is reading it; the status says what happened, for the logs.
        return Response(status_code=499)


# --- operational -------------------------------------------------------------


@router.get("/health", response_model=HealthPayload, tags=["ops"])
def health() -> HealthPayload:
    """Liveness. Answers before the models have loaded."""
    return HealthPayload(status="ok", version=__version__)


@router.get("/ready", response_model=ReadyPayload, tags=["ops"])
def ready(response: Response, services: Services = Depends(get_services)) -> ReadyPayload:
    """Readiness: whether a turn could be served now, and what each dependency says.

    503 while the bank cannot be reached or the service is shutting down. A
    provider whose circuit is open is named in `checks` but leaves the
    service ready: it still answers every turn, only less well.
    """
    is_ready, checks = services.readiness()
    if not is_ready:
        response.status_code = 503
    described = services.describe()
    return ReadyPayload(
        ready=is_ready,
        asr=described["asr"],
        llm=described["llm"],
        tts=described["tts"],
        device=services.device,
        sessions=len(services.sessions),
        checks=checks,
    )


@router.get("/metrics", response_class=PlainTextResponse, tags=["ops"])
def metrics(services: Services = Depends(get_services)) -> str:
    """Prometheus text exposition, hand-rolled.

    The guard hit rate is the number worth watching. A rising one means the
    model is reaching for figures it was not given, which is the failure this
    whole design exists to prevent.
    """
    lines = [
        "# HELP voicebot_turns_total Conversational turns served.",
        "# TYPE voicebot_turns_total counter",
        f"voicebot_turns_total {services.turns}",
        "# HELP voicebot_tool_calls_total Tool invocations by name.",
        "# TYPE voicebot_tool_calls_total counter",
    ]
    for tool, calls in sorted(services.tool_calls.items()):
        lines.append(f'voicebot_tool_calls_total{{tool="{tool}"}} {calls}')
    try:
        guard_hits = str(services.bank.guard_hit_count())
    except Exception:  # the bank is down: the metric is unknown, not the endpoint
        guard_hits = "NaN"
    lines += [
        "# HELP voicebot_guard_hits_total Replies suppressed for stating an unsourced figure.",
        "# TYPE voicebot_guard_hits_total counter",
        f"voicebot_guard_hits_total {guard_hits}",
        "# HELP voicebot_sessions Active conversations.",
        "# TYPE voicebot_sessions gauge",
        f"voicebot_sessions {len(services.sessions)}",
        "# HELP voicebot_turns_in_flight Turns being answered right now.",
        "# TYPE voicebot_turns_in_flight gauge",
        f"voicebot_turns_in_flight {services.in_flight}",
    ]
    if services.audit is not None:
        lines += [
            "# HELP voicebot_audit_queue Turn records waiting to be written.",
            "# TYPE voicebot_audit_queue gauge",
            f"voicebot_audit_queue {services.audit.pending}",
            "# HELP voicebot_audit_dead_letters_total Turn records kept in the dead-letter file.",
            "# TYPE voicebot_audit_dead_letters_total counter",
            f"voicebot_audit_dead_letters_total {services.audit.dead}",
        ]
    lines += metrics_lines()
    return "\n".join(lines) + "\n"


@router.get("/v1/messages", tags=["ops"])
def fallback_messages() -> dict[str, dict[str, str]]:
    """The bot's fallback sentences, in English and Urdu, from messages.yaml.

    The page fetches them once, so what it says when it cannot reach the
    service -- a lost connection, a reply that took too long -- comes from
    the same one file as everything else the bot says when something fails.
    """
    return messages()


# --- single stages, for debugging -------------------------------------------


@router.post("/v1/transcribe", response_model=TranscriptPayload, tags=["stages"])
async def transcribe(
    audio: UploadFile = File(..., description="16-bit PCM WAV"),
    services: Services = Depends(get_services),
) -> TranscriptPayload:
    """Audio to text. The ASR stage on its own."""
    buffer = _read_audio(await audio.read())
    try:
        # A worker thread: recognition blocks, and the event loop is everyone's.
        transcript = await run_in_threadpool(services.asr.transcribe, buffer)
    except ASRUnavailable as exc:
        raise HTTPException(
            status_code=503, detail="Speech recognition is unavailable just now."
        ) from exc
    return TranscriptPayload(
        text=transcript.text,
        confidence=transcript.confidence,
        duration_s=transcript.duration_s,
    )


@router.post("/v1/speak", response_model=AudioPayload, tags=["stages"])
def speak(request: SpeakRequest, services: Services = Depends(get_services)) -> AudioPayload:
    """Text to audio. The TTS stage on its own.

    Returns empty audio when the page is doing the speaking, which is the
    honest answer rather than an error.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Nothing to say.")
    buffer = services.tts.synthesize(request.text)
    return AudioPayload(
        audio_b64=base64.b64encode(encode_wav(buffer)).decode("ascii"),
        sample_rate=buffer.sample_rate,
        duration_s=buffer.duration_s,
    )


# --- a whole turn ------------------------------------------------------------


@router.get("/v1/accounts", response_model=LedgerPayload, tags=["conversation"])
def accounts(services: Services = Depends(get_services)) -> LedgerPayload:
    """The balances and bills, with no turn taken.

    The page calls this once on load so it opens showing real money rather
    than placeholders. Every later figure arrives with the turn that changed
    it, so this never competes with a reply for which number is current.
    """
    return ledger_payload(services.bank)


# What a screen is allowed to run. A screen is not a way to call any tool
# with any arguments: the card buttons need exactly one, and anything else
# belongs to the conversation, where the model chose it.
SCREEN_ACTIONS: frozenset[str] = frozenset({"freeze_card"})


@router.post("/v1/action", response_model=ActionPayload, tags=["screens"])
def action(request: ActionRequest, services: Services = Depends(get_services)) -> ActionPayload:
    """Arm a card action from the cards screen.

    Runs the tool's first phase, which changes nothing and returns the
    bank's own read-back. The screen shows that sentence and settles it
    through /v1/confirm, the same endpoint the spoken Yes button uses --
    one gate, whether the customer clicked or spoke.
    """
    if request.tool not in SCREEN_ACTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"{request.tool!r} cannot be run from a screen.",
        )
    if not request.reason.strip():
        raise HTTPException(status_code=422, detail="Say why, so it can be written down.")

    sid = _session_id(services.sessions, request.session_id)
    session = services.sessions.get(sid)
    session.executor.session_id = sid
    session.executor.turn_index = session.conversation.turn_index

    outcome = session.executor.execute(
        ToolCall(
            id=f"screen-{secrets.token_urlsafe(6)}",
            name=request.tool,
            arguments=dict(request.arguments),
        )
    )
    armed = outcome.result.get("status") == "needs_confirmation"
    summary = str(outcome.result.get("summary") or "")

    if armed:
        # What the customer asked for, in the conversation, before it can be
        # agreed to. Without it the history holds a tool call and its result
        # and nothing anybody said, which a real server answers with 400 --
        # and the typed reason is the only record of why anyway.
        session.conversation.turn_index += 1
        session.conversation.append_user(f"{summary} — {request.reason.strip()}")
        session.conversation.trim()

    services.sessions.put(sid, session)
    error = None if armed else str(outcome.result.get("error") or "That could not be done.")

    # Written down whether it was armed or refused, because what the customer
    # asked for is as much a part of the record as what happened.
    services.record(
        TurnRecord(
            session_id=sid,
            turn_index=session.conversation.turn_index,
            user_text=request.reason.strip(),
            bot_text=summary or (error or ""),
            tool_calls=json.dumps([{"name": request.tool, "arguments": request.arguments}]),
            guard_hit=False,
            suppressed=None,
            latency_ms=0.0,
        )
    )
    if not armed:
        log.info("screen action refused: %s", error)

    return ActionPayload(session_id=sid, armed=armed, summary=summary, error=error)


@router.get("/v1/cards", response_model=CardsPayload, tags=["screens"])
def cards(services: Services = Depends(get_services)) -> CardsPayload:
    """Every card on file, for the cards screen. Four digits each, no more."""
    return cards_payload(services.bank)


@router.get("/v1/monthly", response_model=MonthlyPayload, tags=["screens"])
def monthly(services: Services = Depends(get_services)) -> MonthlyPayload:
    """Every monthly commitment, grouped, for the monthly screen."""
    return monthly_payload(services.bank)


def _card_or_404(services: Services, last4: str) -> Card:
    card = next((c for c in services.bank.cards() if c.last4 == last4), None)
    if card is None:
        raise HTTPException(status_code=404, detail=f"No card ending {last4}.")
    return card


def _checked(services: Services, last4: str, pin: str) -> None:
    """Let a PIN through, or say why not. Nothing here writes it down."""
    try:
        if not services.bank.check_pin(last4, pin):
            # No count, no tries remaining, no hint. Everything that could be
            # said here is something a person guessing would like to know.
            raise HTTPException(status_code=403, detail="That PIN is not right.")
    except BankError as exc:
        # The bank refuses a locked card rather than answering it, so a lock
        # arrives here as an error and not as a wrong PIN.
        raise HTTPException(status_code=423, detail=str(exc)) from exc


@router.post("/v1/cards/{last4}/details", response_model=CardDetailsPayload, tags=["screens"])
def card_details(
    last4: str, request: PinRequest, services: Services = Depends(get_services)
) -> CardDetailsPayload:
    """One card's printed details, once the PIN has been given.

    A POST because it carries a credential and counts a wrong one. Its own
    request rather than part of the card list: a number nobody asked to see
    should not travel with every refresh of the screen.
    """
    card = _card_or_404(services, last4)
    _checked(services, last4, request.pin)
    return card_details_payload(card)


@router.post("/v1/cards/{last4}/pin/check", tags=["screens"])
def check_pin(
    last4: str, request: PinRequest, services: Services = Depends(get_services)
) -> dict[str, str]:
    """Does this PIN open the card? Answers that and nothing else.

    The reset flow asks before it offers a new PIN. Separate from the
    details endpoint on purpose: finding out whether somebody knows the PIN
    should not hand them the card number for asking.
    """
    _card_or_404(services, last4)
    _checked(services, last4, request.pin)
    return {"status": "ok"}


@router.post("/v1/cards/{last4}/pin", tags=["screens"])
def change_pin(
    last4: str, request: PinChangeRequest, services: Services = Depends(get_services)
) -> dict[str, str]:
    """Change a card's PIN, given the one it is now."""
    _card_or_404(services, last4)
    if len(request.new_pin) != PIN_LENGTH or not request.new_pin.isdigit():
        raise HTTPException(
            status_code=422, detail=f"A PIN is {PIN_LENGTH} digits, and nothing else."
        )
    _checked(services, last4, request.current_pin)

    try:
        services.bank.set_pin(last4, request.current_pin, request.new_pin)
    except BankError as exc:  # pragma: no cover - the checks above cover these
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    log.info("pin changed on card %s", last4)
    return {"status": "changed"}


@router.get("/v1/statement", response_model=StatementPayload, tags=["screens"])
def statement(
    year: int | None = None,
    month: int | None = None,
    services: Services = Depends(get_services),
) -> StatementPayload:
    """One calendar month of movements, for the transactions screen.

    Defaults to the month the server is in. A month with nothing in it is an
    empty statement; only a month that is not a month is a bad request.
    """
    today = date.today()
    try:
        return statement_payload(services.bank, year or today.year, month or today.month)
    except BankError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/v1/converse", response_model=TurnPayload, tags=["conversation"])
async def converse(
    request: Request,
    audio: UploadFile = File(..., description="16-bit PCM WAV"),
    session_id: str | None = Form(default=None),
    services: Services = Depends(get_services),
) -> TurnPayload | Response:
    """Audio in, audio out. The whole pipeline, with conversation state."""
    data = await audio.read()
    sid = _session_id(services.sessions, session_id)
    try:
        buffer = _read_audio(data)
    except HTTPException as refused:
        # A recording that cannot be used is still a turn: the customer spoke,
        # and hears a sentence saying so, not an error box.
        detail = refused.detail

        def unusable(session: Session, pipeline: VoicePipeline) -> TurnResult:
            log.info("answering an unusable recording: %s", detail)
            return pipeline.say(say("bad_audio", session.conversation.language), key="bad_audio")

        return await _turn(request, services, sid, unusable)

    return await _turn(
        request,
        services,
        sid,
        lambda session, pipeline: pipeline.converse(buffer, session.conversation),
    )


@router.post("/v1/chat", response_model=TurnPayload, tags=["conversation"])
async def chat(
    http: Request, request: TextRequest, services: Services = Depends(get_services)
) -> TurnPayload | Response:
    """Type instead of speaking, same turn otherwise."""
    sid = _session_id(services.sessions, request.session_id)
    return await _turn(
        http,
        services,
        sid,
        lambda session, pipeline: pipeline.respond_to_text(request.text, session.conversation),
    )


# --- confirmation ------------------------------------------------------------


@router.post("/v1/confirm", response_model=TurnPayload, tags=["conversation"])
async def confirm(
    http: Request, request: ConfirmRequest, services: Services = Depends(get_services)
) -> TurnPayload | Response:
    """Agree to the pending action, and approve it if it moves money.

    This acts on the executor directly, without asking the model whether it
    agrees -- which is what makes a misheard "yes" harmless. The model is then
    told what happened, so its next turn knows the money moved.

    An action that moves money needs the PIN for the account it leaves. The
    first press arrives without one and comes back asking for it; the second
    carries it, and it is checked here against the bank. A wrong PIN is a 403
    and the action stays pending, so the customer can try again without
    having to agree to it a second time.

    The PIN is checked inside the turn, under the conversation's lock, so the
    action it approves cannot be swapped for another between the check and
    the approval -- and a second press of Yes waits for the first.
    """
    pin = request.pin

    def agree(session: Session, pipeline: VoicePipeline) -> TurnResult:
        if pin is not None:
            last4 = session.executor.pin_card()
            if last4 is None:
                # Nothing pending, or pending but not something that moves
                # money. Either way there is no card to check this against,
                # and a PIN with nothing to approve is not approval of anything.
                raise HTTPException(status_code=409, detail="Nothing is waiting on a PIN.")

            # The bank does the checking: it counts a wrong one and locks the
            # card after too many. Only the fact that it was right travels on.
            _checked(services, last4, pin)
            session.executor.pin_accepted()
        return pipeline.resolve_pending(True, session.conversation)

    return await _turn(http, services, _session_id(services.sessions, request.session_id), agree)


@router.post("/v1/cancel", response_model=TurnPayload, tags=["conversation"])
async def cancel(
    http: Request, request: SessionRequest, services: Services = Depends(get_services)
) -> TurnPayload | Response:
    """Decline the pending action."""
    return await _turn(
        http,
        services,
        _session_id(services.sessions, request.session_id),
        lambda session, pipeline: pipeline.resolve_pending(False, session.conversation),
    )


# --- housekeeping ------------------------------------------------------------


@router.post("/v1/reset", tags=["conversation"])
def reset(request: SessionRequest, services: Services = Depends(get_services)) -> dict[str, str]:
    """Put the demo bank back to its opening balances.

    v1 got this for free by throwing away a per-session bank. With one shared
    database, repeated demos would otherwise drain checking until every
    transfer failed for want of funds. The audit log survives.
    """
    services.bank.reset()
    if request.session_id:
        services.sessions.drop(request.session_id)
    return {"status": "reset"}


@router.delete("/v1/session/{session_id}", tags=["conversation"])
def end_session(session_id: str, services: Services = Depends(get_services)) -> dict[str, str]:
    """Forget a conversation, leaving the money alone."""
    services.sessions.drop(session_id)
    return {"status": "ended", "session_id": session_id}
