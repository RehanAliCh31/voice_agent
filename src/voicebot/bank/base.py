"""What a bank can do, and the vocabulary its callers share.

The interface is deliberately narrow: twenty-one operations, each one a thing a
customer can ask for. Anything the dialogue needs that is not on this list is
a sign the dialogue is inventing capability.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta


class BankError(Exception):
    """The request was understood but cannot be carried out.

    Every message is written to be read aloud to the customer, because the
    tool executor passes it straight to the model as a hint.
    """


@dataclass(frozen=True)
class Account:
    name: str
    balance_cents: int


# How many wrong PINs before the card stops accepting them, and for how
# long. Three and five minutes is roughly what a cash machine gives you, and
# it is the difference between a secret and a formality.
PIN_TRIES = 3
PIN_LOCK_MINUTES = 5
PIN_LENGTH = 4


def pin_locked_message() -> str:
    """Said aloud when the PIN has been refused too many times."""
    return (
        f"That PIN has been entered wrongly {PIN_TRIES} times, so it is locked "
        f"for {PIN_LOCK_MINUTES} minutes."
    )


def bad_pin_message() -> str:
    """Said aloud when a new PIN is not a PIN."""
    return f"A PIN is {PIN_LENGTH} digits, and nothing else."


# What a PIN can be asked for. Closed, because each one is a screen the
# page knows how to open: a reason not on this list has nowhere to go.
PIN_PURPOSES: tuple[str, ...] = ("details", "reset", "payment")

# Of those, the ones the model may ask for. "payment" is not among them: the
# gate in front of money asks for that one itself, once there is an agreed
# action to approve. A model asking for it out of the blue would put a keypad
# on screen with nothing behind it to approve.
MODEL_PIN_PURPOSES: tuple[str, ...] = ("details", "reset")

# Bills are paid from checking. Named once, because two places need to agree
# on it: the bank, which takes the money, and the gate that asks for the PIN
# belonging to the account it comes out of.
BILL_ACCOUNT = "checking"


def unknown_purpose_message(purpose: str) -> str:
    """Said aloud when a PIN is asked for something it is not needed for."""
    return (
        f"I don't ask for a PIN to {purpose}. I ask for one to show card "
        f"details, to change the PIN itself, or to let money leave an account."
    )


# What a card can be allowed or refused, one switch each. Closed, because
# each name is a column: a fourth would be a schema change, not a string.
CARD_CONTROLS: tuple[str, ...] = ("ecommerce", "retail", "atm")


def unknown_control_message(control: str) -> str:
    """Said aloud when a card control is named that the card does not have."""
    return f"{control!r} is not a card setting. I can change: {', '.join(CARD_CONTROLS)}."


@dataclass(frozen=True)
class Card:
    last4: str
    account: str
    frozen: bool
    kind: str = "debit"
    holder: str = ""
    # The whole number and the dates printed on it. Shown only on the details
    # screen, never in a reply and never in the audit log: everywhere else
    # four digits is how a card is named.
    number: str = ""
    valid_from: str = ""
    valid_to: str = ""
    ecommerce: bool = True
    retail: bool = True
    atm: bool = True

    @property
    def controls(self) -> dict[str, bool]:
        """The switches, named, so a caller can loop rather than remember."""
        return {name: bool(getattr(self, name)) for name in CARD_CONTROLS}


@dataclass(frozen=True)
class Transaction:
    """One movement on the ledger.

    `cents` keeps its sign, because the sign is the direction and a caller
    that has to infer it will eventually infer it wrong. `spent` names it so
    nobody has to remember which way round it goes.
    """

    id: int
    account: str
    merchant: str
    cents: int
    occurred_at: str

    @property
    def spent(self) -> bool:
        return self.cents < 0


@dataclass(frozen=True)
class Statement:
    """One calendar month of movements, with its totals.

    Both totals are positive: the ledger keeps the sign, and a screen reading
    "Outgoing -Rs. 42,877" has said the same thing twice. Summed here rather
    than in the page, so what is shown and what the bot says are one sum.
    """

    year: int
    month: int
    rows: tuple[Transaction, ...]
    incoming_cents: int
    outgoing_cents: int


@dataclass(frozen=True)
class Dispute:
    """A charge the customer says is wrong, and the case raised for it.

    `amount_cents` is the charge as the ledger holds it, positive, so the
    figure read back is the bank's and not the model's.
    """

    reference: str
    transaction_id: int
    merchant: str
    amount_cents: int
    occurred_at: str
    reason: str
    note: str | None
    status: str
    created_at: str


@dataclass(frozen=True)
class Handoff:
    """A conversation that left the bot for a person.

    Holds no transcript. `session_id` and `turn_index` locate the exchange in
    the turns already logged, so there is one record of what was said rather
    than two that can drift apart.
    """

    reference: str
    session_id: str
    turn_index: int
    reason: str
    note: str | None
    created_at: str


@dataclass(frozen=True)
class Payee:
    """Someone money may be sent to. Never one of the customer's accounts."""

    name: str


# How a monthly commitment is grouped on screen. Closed, because each one
# is a heading somebody reads: a sixth would be a design decision, not a
# string the model happened to produce.
PLAN_CATEGORIES: tuple[str, ...] = (
    "rent",
    "bills",
    "wifi",
    "groceries",
    "subscriptions",
)


@dataclass(frozen=True)
class Bill:
    name: str
    amount_cents: int
    paid: bool
    category: str = "bills"
    # The day of the month it falls due. Capped at 28 by the seed so that
    # every month has one; a bill due on the 31st has no February.
    due_day: int = 1


# The only periods the bot understands. A closed vocabulary rather than free
# text: an unrecognised period used to match no rows and sum to zero, so the
# bot answered "you spent nothing" to a question it had not understood.
PERIODS: tuple[str, ...] = (
    "today",
    "yesterday",
    "this week",
    "last week",
    "this month",
    "last month",
)


# What can be wrong with a charge. Closed, because the reason decides what
# evidence the disputes team asks for, and "other" would mean asking for all
# of it every time.
DISPUTE_REASONS: tuple[str, ...] = (
    "not recognised",
    "never arrived",
    "wrong amount",
    "charged twice",
    "subscription cancelled",
)

# How long after a charge it can still be raised. Card schemes impose a
# window; a bot that ignores it files cases that are thrown out later, after
# the customer has been told theirs was accepted.
DISPUTE_WINDOW_DAYS = 120


def unknown_dispute_reason_message(reason: str) -> str:
    """Said aloud when a dispute names a reason the bank does not file."""
    return (
        f"{reason!r} is not something I can raise a dispute for. "
        f"I can raise: {', '.join(DISPUTE_REASONS)}."
    )


def too_old_to_dispute_message(days: int) -> str:
    """Said aloud when the charge is outside the window."""
    return (
        f"That charge is {days} days old, and a dispute has to be raised "
        f"within {DISPUTE_WINDOW_DAYS} days. A colleague can still look at it."
    )


# Why a conversation left the bot. Closed, because the reason decides who
# picks the case up and how fast -- a fraud report and a complaint do not go
# to the same desk, and free text would route neither.
HANDOFF_REASONS: tuple[str, ...] = (
    "fraud",
    "complaint",
    "hardship",
    "not permitted",
    "customer asked",
    "stuck",
)


def unknown_reason_message(reason: str) -> str:
    """Said aloud when a handoff names a reason the bank does not route."""
    return f"{reason!r} is not a reason I can raise. I can raise: {', '.join(HANDOFF_REASONS)}."


def account_not_payee_message(name: str) -> str:
    """Said aloud when a payment names one of the customer's own accounts.

    Its own message rather than the generic one: this is the mistake that
    actually happened, and the sentence is what tells the model to move the
    money between accounts instead of sending it out of the bank.
    """
    return (
        f"{name} is one of your own accounts, not a payee. "
        "To put money into it, move it between your accounts instead."
    )


def unknown_payee_message(name: str, known: tuple[str, ...]) -> str:
    """Said aloud when a payment names someone the bank has never heard of."""
    return f"I don't have a payee called {name!r}. I can pay: {', '.join(known)}."


def _midnight(moment: datetime) -> datetime:
    return moment.replace(hour=0, minute=0, second=0, microsecond=0)


def _month_start(moment: datetime) -> datetime:
    return _midnight(moment).replace(day=1)


def period_range(period: str, now: datetime) -> tuple[datetime, datetime]:
    """Map a named period onto a half-open `[start, end)` pair.

    Half-open so neighbouring periods cannot both claim the same instant.
    Weeks start on Monday.
    """
    today = _midnight(now)
    day = timedelta(days=1)

    if period == "today":
        return today, today + day
    if period == "yesterday":
        return today - day, today
    if period == "this week":
        monday = today - timedelta(days=today.weekday())
        return monday, monday + timedelta(days=7)
    if period == "last week":
        monday = today - timedelta(days=today.weekday())
        return monday - timedelta(days=7), monday
    if period == "this month":
        start = _month_start(now)
        return start, _month_start(start + timedelta(days=32))
    if period == "last month":
        start = _month_start(now)
        return _month_start(start - day), start

    raise BankError(unknown_period_message(period))


def unknown_period_message(period: str) -> str:
    """Say what went wrong AND what is allowed.

    The model sees this as a tool hint, so a message that does not list the
    options leaves it guessing at a second wrong answer.
    """
    return f"I don't have a period called {period!r}. Try: {', '.join(PERIODS)}."


def format_money(cents: int) -> str:
    """Render integer cents as a readable amount.

    No currency parameter: the bank holds one currency. A parameter nobody
    should pass is an invitation to pass it.
    """
    return f"${cents / 100:,.2f}"


class Bank(ABC):
    """One customer's accounts, cards, bills and history."""

    # --- reads ---------------------------------------------------------------

    @abstractmethod
    def accounts(self) -> tuple[Account, ...]:
        """Every account held."""

    @abstractmethod
    def balance(self, account: str) -> int:
        """Balance in cents. Raises BankError for an unknown account."""

    @abstractmethod
    def cards(self) -> tuple[Card, ...]:
        """Every card on file."""

    @abstractmethod
    def bills(self) -> tuple[Bill, ...]:
        """Every bill on file."""

    @abstractmethod
    def payees(self) -> tuple[Payee, ...]:
        """Everyone money may be sent to."""

    @abstractmethod
    def transactions(
        self,
        limit: int = 5,
        account: str | None = None,
        merchant: str | None = None,
        period: str | None = None,
    ) -> tuple[Transaction, ...]:
        """The most recent movements, newest first.

        The same table `spending` totals, read row by row: this is what
        answers "what was that charge", where the total cannot. Given a
        `period` it covers the same half-open range `spending` uses, so a
        list of rows and a total over them describe the same stretch of time.

        Raises BankError when `account` is not one the customer holds, or
        when `period` is not one of PERIODS.
        """

    @abstractmethod
    def spending(self, period: str, merchant: str | None = None) -> int:
        """Total spent in `period`, as a positive number of cents.

        Raises BankError when `period` is not one of PERIODS.
        """

    # --- writes --------------------------------------------------------------

    @abstractmethod
    def transfer(self, cents: int, payee: str, from_account: str) -> int:
        """Send money to a payee and return the new balance.

        Money leaves the bank. `payee` must be one of `payees()`; an account
        name is refused, because sending to an account is not what this does
        and doing it anyway loses the money.

        Raises BankError for an unknown payee, an unknown account, or
        insufficient funds.
        """

    @abstractmethod
    def move(self, cents: int, from_account: str, to_account: str) -> tuple[int, int]:
        """Move money between the customer's own accounts.

        Returns the two new balances, source first. Debit and credit happen
        together or not at all, so the bank's total never changes.

        Raises BankError if the accounts are the same, either is unknown, the
        amount is not positive, or there is not enough to move.
        """

    @abstractmethod
    def pay_bill(self, bill: str) -> int:
        """Pay a bill in full and return the amount paid."""

    @abstractmethod
    def check_pin(self, last4: str, pin: str, now: datetime | None = None) -> bool:
        """Is this the card's PIN? Counts the failures and locks on PIN_TRIES.

        A correct PIN clears the count. Raises BankError if the card is
        unknown, or if it is already locked -- a locked card is not answered
        with "wrong", because that would be another guess.
        """

    @abstractmethod
    def pin_locked(self, last4: str, now: datetime | None = None) -> bool:
        """Whether the card is refusing PINs for the moment."""

    @abstractmethod
    def set_pin(self, last4: str, current: str, new: str) -> None:
        """Change the PIN, given the one it is now.

        Raises BankError when the current PIN is wrong, the card is locked,
        or the new PIN is not PIN_LENGTH digits.
        """

    @abstractmethod
    def set_card_control(self, last4: str, control: str, enabled: bool) -> Card:
        """Allow or refuse one kind of spending on a card, and return it.

        Raises BankError for an unknown card or a control that is not one of
        CARD_CONTROLS.
        """

    @abstractmethod
    def freeze_card(self, last4: str) -> Card:
        """Freeze a card and return it."""

    @abstractmethod
    def order_card(self, account: str) -> Card:
        """Issue a new card for an account and return it."""

    @abstractmethod
    def deposit(self, cents: int, account: str, source: str = "deposit") -> int:
        """Pay money in and return the new balance.

        Administrative, not conversational. There is deliberately no tool for
        this and there is not meant to be one: an assistant that can credit an
        account on request is not modelling a bank. Money arrives at a branch,
        at a cash machine, or from another bank, and none of those are things
        a voice assistant does.

        It exists so a demo account that has been spent down can be topped up
        from the command line without discarding the audit log. `source` is
        written to the ledger, so even a top-up is a recorded movement rather
        than a balance that grew on its own.
        """

    # --- audit ---------------------------------------------------------------

    @abstractmethod
    def statement(self, year: int, month: int) -> Statement:
        """Every movement in one calendar month, newest first, with totals.

        A month holding nothing is an empty statement, not an error. Raises
        BankError only when the month is not a month.
        """

    @abstractmethod
    def transaction(self, transaction_id: int) -> Transaction | None:
        """One ledger row by id, or None. Needed to read a charge back before
        a dispute is filed against it."""

    @abstractmethod
    def open_dispute(
        self,
        transaction_id: int,
        reason: str,
        note: str | None = None,
        now: datetime | None = None,
    ) -> Dispute:
        """File a dispute against one charge and return the case.

        Files only. The returned case is always open, no money moves, and
        nothing here decides whether the customer is right.

        Raises BankError when the transaction is unknown, is money arriving
        rather than a charge, has already been disputed, is older than
        DISPUTE_WINDOW_DAYS, or the reason is not one of DISPUTE_REASONS.
        """

    @abstractmethod
    def disputes(self) -> tuple[Dispute, ...]:
        """Every dispute on file, newest first."""

    @abstractmethod
    def open_handoff(
        self,
        session_id: str,
        turn_index: int,
        reason: str,
        note: str | None = None,
    ) -> Handoff:
        """Record that this conversation needs a person, and return the case.

        Raises BankError when `reason` is not one of HANDOFF_REASONS.
        """

    @abstractmethod
    def handoffs(self, session_id: str | None = None) -> tuple[Handoff, ...]:
        """Open cases, newest first, optionally for one conversation."""

    @abstractmethod
    def log_turn(
        self,
        session_id: str,
        turn_index: int,
        user_text: str,
        bot_text: str,
        tool_calls: str | None,
        guard_hit: bool,
        suppressed: str | None,
        latency_ms: float,
    ) -> None:
        """Record one exchange."""

    @abstractmethod
    def turn_count(self) -> int:
        """How many turns have been logged."""

    @abstractmethod
    def guard_hit_count(self) -> int:
        """How many logged turns had their sentence suppressed."""

    @abstractmethod
    def reset(self) -> None:
        """Restore the seeded account state, leaving the audit log alone."""
