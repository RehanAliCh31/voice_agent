"""Request and response bodies for the HTTP API.

Separate from `voicebot.schemas` on purpose. Those dataclasses are the
pipeline's internal contract; these are the *published* one. Serving the
internals directly would mean any refactor is a breaking API change, and the
front end would have to follow every rename.
"""

from __future__ import annotations

import base64
from datetime import date
from typing import Any

from pydantic import BaseModel, Field

from voicebot.bank.base import Bank, format_money
from voicebot.pipeline.audio_io import encode_wav
from voicebot.pipeline.orchestrator import TurnResult


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
    accounts: list[AccountPayload] = Field(default_factory=list)
    bills: list[BillPayload] = Field(default_factory=list)
    # The reference of a case this turn opened, or null. The page shows the
    # conversation has left the bot without having to read the reply for it.
    handed_off: str | None = None
    # The reference of a dispute this turn filed, or null.
    dispute_filed: str | None = None
    latency_ms: dict[str, float] = Field(default_factory=dict)
    # Null when the page is doing the speaking. Nullable rather than absent so
    # a client can tell "no audio" from "field missing".
    audio_b64: str | None = None
    speak_client_side: bool = False
    audio_sample_rate: int = 16_000


class TextRequest(BaseModel):
    """Type instead of speaking. Used by the front end's fallback input."""

    text: str
    session_id: str | None = None


class SessionRequest(BaseModel):
    """Act on a session without saying anything -- confirm, cancel, reset."""

    session_id: str | None = None


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
        accounts=_accounts(bank),
        bills=_bills(bank),
        handed_off=result.agent.handed_off,
        dispute_filed=result.agent.dispute_filed,
        latency_ms={key: round(value, 2) for key, value in result.latency.as_dict().items()},
        audio_b64=audio_b64,
        speak_client_side=speaks_client_side,
        audio_sample_rate=result.audio.sample_rate,
    )
