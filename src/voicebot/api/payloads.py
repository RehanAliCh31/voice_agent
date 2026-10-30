"""Request and response bodies for the HTTP API.

Separate from `voicebot.schemas` on purpose. Those dataclasses are the
pipeline's internal contract; these are the *published* one. Serving the
internals directly would mean any refactor is a breaking API change, and the
front end would have to follow every rename.
"""

from __future__ import annotations

import base64
import secrets
from collections.abc import Callable
from datetime import date
from typing import Any, TypeVar

from pydantic import BaseModel, Field

from voicebot.bank.base import PLAN_CATEGORIES, Bank, Card, format_money
from voicebot.errors import DatabaseError, current_call_id
from voicebot.pipeline.audio_io import encode_wav
from voicebot.pipeline.orchestrator import TurnResult
from voicebot.utils.observability import report

T = TypeVar("T")


class TranscriptPayload(BaseModel):
    text: str
    confidence: float
    duration_s: float


class ToolCallPayload(BaseModel):
    """One tool the model ran, with what it asked for and what came back.

    This is what the inspector panel shows in place of v1's intent, slots and
    gate score: the actual work done, rather than a classifier's opinion.
    """

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)


class AccountPayload(BaseModel):
    """One account as the page shows it: the figure and the words for it."""

    name: str
    balance: str
    balance_cents: int


class BillPayload(BaseModel):
    name: str
    amount: str
    amount_cents: int
    paid: bool


class TurnPayload(BaseModel):
    """Everything one turn produced: what was heard, said, done, and changed."""

    session_id: str
    transcript: TranscriptPayload
    text: str
    reply: str
    tool_calls: list[ToolCallPayload] = Field(default_factory=list)
    guard_hit: bool = False
    # True while an action has been read back and not yet agreed to. The page
    # shows its Yes/Cancel buttons from this.
    awaiting_confirmation: bool = False
    # The account state as it stands AFTER this turn. It travels with the
    # reply rather than in a request of its own, so the balances on screen and
    # the figures the bot just spoke always come from the same answer. Fetched
    # separately they could disagree for a moment, which in a banking
    # interface is worse than being slow.
    # None when the ledger could not be read just now: the page then keeps
    # what it is showing, rather than clearing it as if the accounts were gone.
    accounts: list[AccountPayload] | None = Field(default_factory=list)
    bills: list[BillPayload] | None = Field(default_factory=list)
    # The reference of a case this turn opened, or null. The page shows the
    # conversation has left the bot without having to read the reply for it.
    handed_off: str | None = None
    # The reference of a dispute this turn filed, or null.
    dispute_filed: str | None = None
    # The keypad this turn asked the page to open, or null. It names a card
    # and a reason; the PIN itself never travels in either direction.
    pin_request: dict[str, str] | None = None
    latency_ms: dict[str, float] = Field(default_factory=dict)
    # Null when the page is doing the speaking. Nullable rather than absent so
    # a client can tell "no audio" from "field missing".
    audio_b64: str | None = None
    speak_client_side: bool = False
    audio_sample_rate: int = 16_000
    # How many more pieces of the reply's audio follow audio_b64, and the id
    # to ask for them by: GET /v1/sessions/{session_id}/audio/{audio_id}/{n},
    # n from 1. Each is ready a moment after the one before.
    audio_more: int = 0
    audio_id: str | None = None
    # When the reply is one of the bot's own fallback sentences, which one --
    # and the call the turn belongs to, for anyone reading the logs after.
    fallback: str | None = None
    call_id: str | None = None


class ActionRequest(BaseModel):
    """One tool run from a screen rather than from the conversation.

    `reason` is required and written to the audit log. A button that asks why
    and then throws the answer away is theatre.
    """

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str = Field(min_length=1)
    session_id: str | None = None


class ActionPayload(BaseModel):
    """What the screen needs to show next: the read-back, or the refusal.

    No token. The gate is settled through /v1/confirm, which acts on the
    pending action directly, so nothing the screen holds could be replayed.
    """

    session_id: str
    armed: bool
    summary: str = ""
    error: str | None = None


class TextRequest(BaseModel):
    """Type instead of speaking. Used by the front end's fallback input."""

    text: str
    session_id: str | None = None


class SessionRequest(BaseModel):
    """Act on a session without saying anything -- confirm, cancel, reset."""

    session_id: str | None = None


class ConfirmRequest(SessionRequest):
    """Agree to the pending action, and approve it if it moves money.

    The PIN is optional because the first press does not carry one: the
    answer to that press is what tells the page to ask for it. It is checked
    against the bank here and goes no further -- not into the conversation,
    not past the executor, and not into the audit log.
    """

    pin: str | None = None


class SpeakRequest(BaseModel):
    text: str


class AudioPayload(BaseModel):
    audio_b64: str
    sample_rate: int
    duration_s: float


class HealthPayload(BaseModel):
    status: str
    version: str


class ReadyPayload(BaseModel):
    ready: bool
    asr: str
    llm: str
    tts: str
    device: str
    sessions: int
    # What each dependency says: the database, every provider's circuit, the
    # audit queue, whether the service is shutting down.
    checks: dict[str, str] = Field(default_factory=dict)


class StatementRowPayload(BaseModel):
    """One movement as the transactions screen draws it.

    The amount is positive and `direction` says which way it went. The ledger
    keeps the sign; a row reading "-$7.60" under an Outgoing heading would be
    saying it twice.
    """

    id: int
    merchant: str
    account: str
    amount: str
    amount_cents: int
    direction: str
    occurred_at: str
    # "merchant" when money met the outside world, "transfer" when it only
    # changed sides between the customer's own accounts. The screen words the
    # two differently, and only the server knows which names are accounts.
    kind: str


class StatementPayload(BaseModel):
    """One calendar month, and the balance it ended at."""

    year: int
    month: int
    month_label: str
    total_balance: str
    total_balance_cents: int
    incoming: str
    incoming_cents: int
    outgoing: str
    outgoing_cents: int
    rows: list[StatementRowPayload] = Field(default_factory=list)


class PlanPayload(BaseModel):
    """One monthly commitment as the monthly screen draws it."""

    name: str
    category: str
    amount: str
    amount_cents: int
    due_day: int
    due_label: str
    paid: bool


class PlanGroupPayload(BaseModel):
    """One heading on the monthly screen, and what falls under it."""

    category: str
    label: str
    total: str
    total_cents: int
    plans: list[PlanPayload] = Field(default_factory=list)


class MonthlyPayload(BaseModel):
    """Everything that leaves this month, and how much of it has gone."""

    month_label: str
    committed: str
    committed_cents: int
    paid: str
    paid_cents: int
    remaining: str
    remaining_cents: int
    groups: list[PlanGroupPayload] = Field(default_factory=list)


class CardControlPayload(BaseModel):
    """One switch on a card, named for the page as well as for the model."""

    name: str
    label: str
    detail: str
    enabled: bool


class CardPayload(BaseModel):
    """One card as the cards screen draws it.

    `masked` is built from the four digits the bank holds. The rest were
    never stored, so what is drawn is dots rather than a number pretending
    to be real.
    """

    last4: str
    masked: str
    account: str
    holder: str
    kind: str
    frozen: bool
    controls: list[CardControlPayload] = Field(default_factory=list)


class PinRequest(BaseModel):
    """A PIN on its way to be checked, and no further.

    It is never logged, never answered back, and never put in the
    conversation. The only thing done with it is a comparison.
    """

    pin: str = Field(min_length=1)


class PinChangeRequest(BaseModel):
    current_pin: str = Field(min_length=1)
    new_pin: str = Field(min_length=1)


class CardDetailsPayload(BaseModel):
    """What a card screen shows about one card, on a deliberate request.

    `security_code` is minted here and kept nowhere. It is different on every
    reading, which is the honest shape for a value the bank does not hold --
    and why the screen counts down rather than leaving it up.
    """

    last4: str
    holder: str
    card_type: str
    number: str
    status: str
    valid_from: str
    valid_to: str
    security_code: str
    expires_in_seconds: int


class CardsPayload(BaseModel):
    cards: list[CardPayload] = Field(default_factory=list)


class LedgerPayload(BaseModel):
    """The accounts and bills on their own, with no turn attached.

    The page needs balances before the first word is spoken. That opening
    paint is a read, so it gets a read's endpoint rather than an empty turn,
    which would reach the model and bank a turn nobody had.
    """

    accounts: list[AccountPayload] = Field(default_factory=list)
    bills: list[BillPayload] = Field(default_factory=list)


def statement_payload(bank: Bank, year: int, month: int) -> StatementPayload:
    """One month of movements, with the totals the screen puts above them."""
    statement = bank.statement(year, month)
    accounts = bank.accounts()
    held = sum(account.balance_cents for account in accounts)
    own = {account.name for account in accounts}
    return StatementPayload(
        year=statement.year,
        month=statement.month,
        # Named by the server so the heading and the rows cannot disagree
        # about which month is on screen.
        month_label=date(statement.year, statement.month, 1).strftime("%B %Y"),
        total_balance=format_money(held),
        total_balance_cents=held,
        incoming=format_money(statement.incoming_cents),
        incoming_cents=statement.incoming_cents,
        outgoing=format_money(statement.outgoing_cents),
        outgoing_cents=statement.outgoing_cents,
        rows=[
            StatementRowPayload(
                id=row.id,
                merchant=row.merchant,
                account=row.account,
                amount=format_money(abs(row.cents)),
                amount_cents=abs(row.cents),
                direction="out" if row.spent else "in",
                occurred_at=row.occurred_at,
                kind="transfer" if row.merchant in own else "merchant",
            )
            for row in statement.rows
        ],
    )


# What each switch is called on screen, and the line under it. Kept beside
# the payload rather than in the page, so the words a customer reads and the
# words the model is given for the same switch come from one place.
_CONTROL_WORDS: dict[str, tuple[str, str]] = {
    "ecommerce": ("Online purchases", "Allow this card to be used online"),
    "retail": ("Shop purchases", "Allow this card to be used in person"),
    "atm": ("Cash machines", "Allow this card to withdraw cash"),
}


# The heading each category gets. Beside the payload rather than in the
# page, for the same reason the card controls are: the words a customer
# reads come from one place.
_CATEGORY_WORDS: dict[str, str] = {
    "rent": "Rent",
    "bills": "Bills",
    "wifi": "Internet",
    "groceries": "Groceries",
    "subscriptions": "Subscriptions",
}


def _ordinal(day: int) -> str:
    """ "the 1st", "the 22nd" -- said the way a date is said aloud."""
    if 11 <= day % 100 <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{day}{suffix}"


def monthly_payload(bank: Bank) -> MonthlyPayload:
    """Every monthly commitment, grouped, with what is left to pay.

    Grouped here rather than in the page so the subtotals under each heading
    and the figure at the top are one arithmetic, and so the order of the
    headings is the bank's rather than whatever order rows arrived in.
    """
    bills = bank.bills()
    committed = sum(bill.amount_cents for bill in bills)
    paid = sum(bill.amount_cents for bill in bills if bill.paid)

    groups = []
    for category in PLAN_CATEGORIES:
        under = [bill for bill in bills if bill.category == category]
        if not under:
            continue
        groups.append(
            PlanGroupPayload(
                category=category,
                label=_CATEGORY_WORDS.get(category, category.title()),
                total=format_money(sum(bill.amount_cents for bill in under)),
                total_cents=sum(bill.amount_cents for bill in under),
                plans=[
                    PlanPayload(
                        name=bill.name,
                        category=bill.category,
                        amount=format_money(bill.amount_cents),
                        amount_cents=bill.amount_cents,
                        due_day=bill.due_day,
                        due_label=f"Due the {_ordinal(bill.due_day)}",
                        paid=bill.paid,
                    )
                    for bill in under
                ],
            )
        )

    return MonthlyPayload(
        month_label=date.today().strftime("%B %Y"),
        committed=format_money(committed),
        committed_cents=committed,
        paid=format_money(paid),
        paid_cents=paid,
        remaining=format_money(committed - paid),
        remaining_cents=committed - paid,
        groups=groups,
    )


# How long the details stay on screen. Long enough to copy a number off,
# short enough that a phone left on a desk is not showing one.
DETAILS_SECONDS = 120


def card_details_payload(card: Card) -> CardDetailsPayload:
    """One card's details, with a code made for this viewing only."""
    return CardDetailsPayload(
        last4=card.last4,
        holder=card.holder,
        card_type=card.kind.title(),
        # Grouped in fours, the way it is printed and the way it is read out.
        number=" ".join(card.number[i : i + 4] for i in range(0, len(card.number), 4)),
        status="Frozen" if card.frozen else "Active",
        valid_from=card.valid_from,
        valid_to=card.valid_to,
        security_code=f"{secrets.randbelow(1000):03d}",
        expires_in_seconds=DETAILS_SECONDS,
    )


def cards_payload(bank: Bank) -> CardsPayload:
    """Every card on file, for the cards screen."""
    return CardsPayload(
        cards=[
            CardPayload(
                last4=card.last4,
                masked=f"•••• •••• •••• {card.last4}",
                account=card.account,
                holder=card.holder,
                kind=card.kind,
                frozen=card.frozen,
                controls=[
                    CardControlPayload(
                        name=name,
                        label=_CONTROL_WORDS[name][0],
                        detail=_CONTROL_WORDS[name][1],
                        enabled=enabled,
                    )
                    for name, enabled in card.controls.items()
                ],
            )
            for card in bank.cards()
        ]
    )


def _accounts(bank: Bank | None) -> list[AccountPayload]:
    if bank is None:
        return []
    return [
        AccountPayload(
            name=a.name, balance=format_money(a.balance_cents), balance_cents=a.balance_cents
        )
        for a in bank.accounts()
    ]


def _bills(bank: Bank | None) -> list[BillPayload]:
    if bank is None:
        return []
    return [
        BillPayload(
            name=b.name,
            amount=format_money(b.amount_cents),
            amount_cents=b.amount_cents,
            paid=b.paid,
        )
        for b in bank.bills()
    ]


def ledger_payload(bank: Bank) -> LedgerPayload:
    """The ledger as it stands now. Same shape a turn carries, same source."""
    return LedgerPayload(accounts=_accounts(bank), bills=_bills(bank))


def turn_payload(
    session_id: str,
    result: TurnResult,
    speaks_client_side: bool,
    awaiting_confirmation: bool = False,
    bank: Bank | None = None,
) -> TurnPayload:
    """Convert a pipeline result into the published shape.

    Audio is base64-encoded WAV inside the JSON rather than a second request
    for a binary body. One round trip per turn keeps the client simple, and a
    reply is a few hundred kilobytes. With browser speech there is no audio at
    all, and `speak_client_side` tells the page to say the words itself.
    """
    # A voice that failed is never handed to the browser's own: it may be
    # the other gender, and has no Urdu. The pipeline has already put the
    # bot's own recording of "my voice isn't working" in its place if it
    # could; failing that the reply is on the screen, silent.
    audio_b64 = None
    if len(result.audio.samples):
        audio_b64 = base64.b64encode(encode_wav(result.audio)).decode("ascii")

    # The token gates money. There is no reason to publish it, even
    # same-origin to the session that owns it.
    calls = [
        ToolCallPayload(
            name=call.name,
            arguments={k: v for k, v in call.arguments.items() if k != "confirmation_token"},
            result={k: v for k, v in outcome.items() if k != "token"},
        )
        for call, outcome in zip(result.agent.tool_calls, result.agent.tool_results, strict=False)
    ]

    return TurnPayload(
        session_id=session_id,
        transcript=TranscriptPayload(
            text=result.transcript.text,
            confidence=result.transcript.confidence,
            duration_s=result.transcript.duration_s,
        ),
        text=result.text,
        reply=result.agent.text,
        tool_calls=calls,
        guard_hit=result.agent.guard_hit,
        awaiting_confirmation=awaiting_confirmation,
        accounts=_read(_accounts, bank),
        bills=_read(_bills, bank),
        handed_off=result.agent.handed_off,
        dispute_filed=result.agent.dispute_filed,
        pin_request=result.agent.pin_request,
        latency_ms={key: round(value, 2) for key, value in result.latency.as_dict().items()},
        audio_b64=audio_b64,
        speak_client_side=speaks_client_side,
        audio_sample_rate=result.audio.sample_rate,
        audio_more=result.rest.count if result.rest is not None else 0,
        audio_id=result.rest.id if result.rest is not None else None,
        fallback=result.agent.fallback,
        call_id=current_call_id() or session_id,
    )


def _read(read: Callable[[Bank | None], list[T]], bank: Bank | None) -> list[T] | None:
    """The ledger for the page, or None when the bank cannot be read just now.

    The turn has already happened by the time this runs. A ledger that fails
    to load must not turn a finished turn -- money moved, perhaps -- into an
    error the customer reads as "it didn't work, try again".
    """
    try:
        return read(bank)
    except Exception as exc:  # the turn is done; the page keeps its last figures
        report(DatabaseError(f"the ledger could not be read: {type(exc).__name__}", cause=exc))
        return None
