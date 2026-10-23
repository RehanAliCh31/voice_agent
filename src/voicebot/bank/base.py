"""What a bank can do, and the vocabulary its callers share.

The interface is deliberately narrow: eleven operations, each one a thing a
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


@dataclass(frozen=True)
class Card:
    last4: str
    account: str
    frozen: bool


@dataclass(frozen=True)
class Transaction:
    """One movement on the ledger.

    `cents` keeps its sign, because the sign is the direction and a caller
    that has to infer it will eventually infer it wrong. `spent` names it so
    nobody has to remember which way round it goes.
    """

    account: str
    merchant: str
    cents: int
    occurred_at: str

    @property
    def spent(self) -> bool:
        return self.cents < 0


@dataclass(frozen=True)
class Bill:
    name: str
    amount_cents: int
    paid: bool


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


# Where money paid in is allowed to have come from. Closed for the same
# reason PERIODS is: free text would let the bot record a deposit from a
# source the bank has no notion of, and money that arrives from nowhere in
# particular is the thing a bank ledger exists to rule out.
DEPOSIT_SOURCES: tuple[str, ...] = (
    "cheque",
    "cash",
    "transfer in",
)


def unknown_source_message(source: str) -> str:
    """Said aloud when a deposit names a source the bank does not take."""
    allowed = ", ".join(DEPOSIT_SOURCES)
    return f"I can't take a deposit by {source}. I can record {allowed}."


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
    def transactions(
        self,
        limit: int = 5,
        account: str | None = None,
        merchant: str | None = None,
    ) -> tuple[Transaction, ...]:
        """The most recent movements, newest first.

        The same table `spending` totals, read row by row: this is what
        answers "what was that charge", where the total cannot.

        Raises BankError when `account` is not one the customer holds.
        """

    @abstractmethod
    def spending(self, period: str, merchant: str | None = None) -> int:
        """Total spent in `period`, as a positive number of cents.

        Raises BankError when `period` is not one of PERIODS.
        """

    # --- writes --------------------------------------------------------------

    @abstractmethod
    def transfer(self, cents: int, payee: str, from_account: str) -> int:
        """Move money out and return the new balance."""

    @abstractmethod
    def pay_bill(self, bill: str) -> int:
        """Pay a bill in full and return the amount paid."""

    @abstractmethod
    def freeze_card(self, last4: str) -> Card:
        """Freeze a card and return it."""

    @abstractmethod
    def order_card(self, account: str) -> Card:
        """Issue a new card for an account and return it."""

    @abstractmethod
    def deposit(self, cents: int, account: str, source: str = "deposit") -> int:
        """Pay money in and return the new balance.

        `source` says where the money came from and is written to the ledger,
        so a deposit is a recorded movement rather than a balance that grew on
        its own. The conversational path accepts only DEPOSIT_SOURCES; the
        command line passes its own label when topping up a demo account.
        """

    # --- audit ---------------------------------------------------------------

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
