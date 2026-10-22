"""A deterministic in-memory bank.

Nothing here talks to a network. The state is seeded and reproducible, so a
demo and an integration test see exactly the same balances, and a failing test
can be rerun to the same numbers.

Money is integer cents everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class BankError(Exception):
    """The request was understood but cannot be carried out."""


@dataclass
class Card:
    last4: str
    frozen: bool = False


@dataclass
class Transaction:
    merchant: str
    cents: int
    period: str


@dataclass
class MockBank:
    """Seeded account state. Construct one per session."""

    balances: dict[str, int] = field(
        default_factory=lambda: {"checking": 248_350, "savings": 1_902_000}
    )
    cards: list[Card] = field(default_factory=lambda: [Card(last4="4417"), Card(last4="9021")])
    transactions: list[Transaction] = field(
        default_factory=lambda: [
            Transaction("amazon", 8_499, "last month"),
            Transaction("tesco", 5_210, "last month"),
            Transaction("netflix", 1_099, "this week"),
            Transaction("uber", 2_340, "this week"),
            Transaction("starbucks", 760, "yesterday"),
        ]
    )
    bills: dict[str, int] = field(
        default_factory=lambda: {
            "electricity": 7_250,
            "water": 3_100,
            "phone": 2_499,
            "internet": 4_000,
            "credit card": 21_500,
        }
    )

    # --- account lookups -----------------------------------------------------

    def knows_account(self, text: str) -> bool:
        """True when `text` names an account this bank actually holds.

        Asking before converting lets a caller treat an unrecognised name as
        no name at all, which is almost always the better reply.
        """
        lowered = text.lower()
        return any(word in lowered for word in ("sav", "check", "current", "joint"))

    def knows_bill(self, text: str) -> bool:
        """True when `text` names a bill on file."""
        return text.lower() in self.bills

    def knows_period(self, text: str) -> bool:
        """True when `text` names a period the statement covers."""
        return any(t.period == text.lower() for t in self.transactions)

    def normalise_account(self, text: str) -> str:
        """Map a spoken account name onto a known account."""
        lowered = text.lower()
        if "sav" in lowered:
            return "savings"
        if "check" in lowered or "current" in lowered or "joint" in lowered:
            return "checking"
        raise BankError(f"I don't recognise the account {text!r}.")

    def balance(self, account: str) -> int:
        key = self.normalise_account(account)
        return self.balances[key]

    # --- actions -------------------------------------------------------------

    def transfer(self, cents: int, payee: str, account: str = "checking") -> int:
        """Move money out and return the new balance."""
        if cents <= 0:
            raise BankError("The amount has to be more than zero.")
        key = self.normalise_account(account)
        if self.balances[key] < cents:
            raise BankError(
                f"There isn't enough in {key}. The balance is {format_money(self.balances[key])}."
            )
        self.balances[key] -= cents
        return self.balances[key]

    def pay_bill(self, bill_type: str) -> int:
        """Pay a bill in full and return the amount paid."""
        key = bill_type.lower()
        if key not in self.bills:
            raise BankError(f"I don't have a {bill_type} bill on file.")
        amount = self.bills[key]
        if self.balances["checking"] < amount:
            raise BankError("There isn't enough in checking to cover that bill.")
        self.balances["checking"] -= amount
        self.bills[key] = 0
        return amount

    def freeze_card(self, last4: str) -> Card:
        for card in self.cards:
            if card.last4 == last4:
                card.frozen = True
                return card
        raise BankError(f"I can't find a card ending {last4}.")

    def spending(self, period: str, merchant: str | None = None) -> int:
        """Total spend in a period, optionally for one merchant."""
        matches = [t for t in self.transactions if t.period == period.lower()]
        if merchant:
            matches = [t for t in matches if t.merchant == merchant.lower()]
        return sum(t.cents for t in matches)


def format_money(cents: int, currency: str = "USD") -> str:
    """Render cents as a readable amount."""
    symbol = {"USD": "$", "GBP": "£", "EUR": "€", "PKR": "Rs "}.get(currency, "")
    return f"{symbol}{cents / 100:,.2f}"
