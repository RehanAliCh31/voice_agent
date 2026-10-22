"""Runs a tool against the bank, and owns the confirmation gate.

The gate is deliberately here and not in the prompt. A model can be talked out
of an instruction; it cannot be talked out of a function that refuses to act
without a token it has not been given.

Nothing in this module raises at its caller. Every failure — a wrong type, an
unknown tool, a bank refusal — comes back as `{"error", "hint"}`, because the
model can recover from a sentence and cannot recover from an exception. "There
isn't enough in checking" is something the bot should be able to explain.
"""

from __future__ import annotations

import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from voicebot.bank.base import Bank, BankError, format_money
from voicebot.llm.base import ToolCall
from voicebot.llm.tools import CONFIRMING
from voicebot.utils.logging import get_logger

log = get_logger("agent.executor")

# How long a read-back summary stays valid. Long enough for a person to think
# about it, short enough that a forgotten confirmation cannot be completed by
# a later, unrelated turn.
DEFAULT_TTL_S = 300.0


@dataclass
class PendingAction:
    """An action that has been read back to the customer but not performed."""

    tool: str
    arguments: dict[str, Any]
    token: str
    summary: str
    created_at: float

    def expired(self, ttl_s: float, now: float) -> bool:
        # >= so that a ttl of zero means "no validity at all" rather than
        # "valid until the clock happens to tick".
        return now - self.created_at >= ttl_s


def _error(message: str, hint: str) -> dict[str, Any]:
    return {"error": message, "hint": hint}


def _require_positive_int(value: Any, field: str) -> int:
    """Accept a whole number of cents and nothing else.

    `isinstance(True, int)` is true in Python, so booleans are rejected
    explicitly. A float is rejected rather than rounded: a model that sends
    50.5 has misunderstood the unit, and rounding would hide that behind a
    transfer of the wrong amount.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{field} must be a whole number of cents, not {type(value).__name__}. "
            "5000 means $50.00."
        )
    if value <= 0:
        raise ValueError(f"{field} must be greater than zero.")
    return value


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string.")
    return value.strip()


class ToolExecutor:
    """Runs tools for one conversation, and holds its one pending action."""

    def __init__(self, bank: Bank, ttl_s: float = DEFAULT_TTL_S) -> None:
        self.bank = bank
        self.ttl_s = ttl_s
        self.pending: PendingAction | None = None

    # --- the public surface --------------------------------------------------

    def execute(self, tool_call: ToolCall) -> dict[str, Any]:
        """Run one tool call, or ask for confirmation first."""
        handler = self._handlers().get(tool_call.name)
        if handler is None:
            return _error(
                f"There is no tool called {tool_call.name!r}.",
                f"Available tools: {', '.join(sorted(self._handlers()))}.",
            )

        arguments = {k: v for k, v in tool_call.arguments.items() if k != "confirmation_token"}
        token = tool_call.arguments.get("confirmation_token")

        try:
            if tool_call.name in CONFIRMING:
                return self._gated(tool_call.name, arguments, token, handler)
            return handler(arguments)
        except (ValueError, KeyError) as exc:
            return _error(str(exc), "Check the tool's parameters and try again.")
        except BankError as exc:
            # The bank writes its refusals to be read aloud, so the message is
            # the hint: it says both what went wrong and what is allowed.
            return _error(str(exc), str(exc))

    def confirm(self) -> dict[str, Any]:
        """Perform the pending action. Called by the Yes button, not the model."""
        pending = self.pending
        if pending is None:
            return _error(
                "There's nothing waiting to be confirmed.",
                "No action was pending, so nothing happened.",
            )
        self.pending = None
        try:
            return self._handlers()[pending.tool](pending.arguments)
        except (ValueError, KeyError) as exc:
            return _error(str(exc), "The pending action could not be completed.")
        except BankError as exc:
            return _error(str(exc), str(exc))

    def cancel(self) -> dict[str, Any]:
        """Discard the pending action. Called by the Cancel button."""
        pending = self.pending
        self.pending = None
        if pending is None:
            return {"status": "cancelled", "detail": "There was nothing pending."}
        return {"status": "cancelled", "detail": f"Cancelled: {pending.summary}"}

    # --- the gate ------------------------------------------------------------

    def _gated(
        self,
        name: str,
        arguments: dict[str, Any],
        token: Any,
        handler: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        """Two-phase. Validate first, so a bad request never becomes pending."""
        validated = self._validate(name, arguments)
        pending = self.pending

        if token is not None:
            if pending is None:
                return _error(
                    "There's nothing waiting to be confirmed.",
                    "Call this tool without a confirmation_token first, read the "
                    "summary back to the customer, and only then confirm.",
                )
            if pending.expired(self.ttl_s, time.monotonic()):
                self.pending = None
                return _error(
                    "That confirmation has expired.",
                    "Ask the customer again, starting from a fresh call.",
                )
            if token != pending.token:
                # A token we never issued. Refuse plainly and keep the real
                # pending action alive, so the customer can still agree to the
                # thing that was actually read back to them.
                log.warning("confirmation refused for %s: unrecognised token", name)
                return _error(
                    "That confirmation code doesn't match anything I'm waiting on.",
                    "Call this tool without a confirmation_token to get a fresh "
                    "summary, read it back, and confirm that one.",
                )
            if pending.tool != name or pending.arguments != validated:
                # A valid token against different arguments: the customer
                # changed their mind mid-confirmation. "Actually make it
                # eighty" has to be agreed to on its own terms, not slipped
                # through under the old agreement.
                log.info("arguments changed under confirmation for %s; re-confirming", name)
                return self._begin(name, validated)
            self.pending = None
            return handler(validated)

        return self._begin(name, validated)

    def _begin(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Record what we are about to do, and say it back. Nothing happens yet."""
        summary = self._summarise(name, arguments)
        self.pending = PendingAction(
            tool=name,
            arguments=arguments,
            token=secrets.token_urlsafe(8),
            summary=summary,
            created_at=time.monotonic(),
        )
        return {"status": "needs_confirmation", "token": self.pending.token, "summary": summary}

    def _validate(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Check a confirming tool's arguments. Raises ValueError with a hint."""
        if name == "transfer_money":
            return {
                "amount_cents": _require_positive_int(
                    arguments.get("amount_cents"), "amount_cents"
                ),
                "payee": _require_text(arguments.get("payee"), "payee"),
                "from_account": _require_text(arguments.get("from_account"), "from_account"),
            }
        if name == "pay_bill":
            return {"bill": _require_text(arguments.get("bill"), "bill")}
        if name == "freeze_card":
            return {
                "last4": _require_text(arguments.get("last4"), "last4"),
                "report_fraud": bool(arguments.get("report_fraud", False)),
            }
        if name == "order_card":
            return {"account": _require_text(arguments.get("account"), "account")}
        raise KeyError(f"{name} is not a confirming tool")

    def _summarise(self, name: str, arguments: dict[str, Any]) -> str:
        """The sentence the customer agrees to. Built from values, never generated."""
        if name == "transfer_money":
            amount = format_money(arguments["amount_cents"])
            return f"Transfer {amount} to {arguments['payee']} from {arguments['from_account']}"
        if name == "pay_bill":
            return f"Pay your {arguments['bill']} bill in full"
        if name == "freeze_card":
            card = f"the card ending {arguments['last4']}"
            if arguments.get("report_fraud"):
                return f"Report {card} as fraudulent and freeze it"
            return f"Freeze {card}"
        if name == "order_card":
            return f"Order a new card for {arguments['account']}"
        return "Go ahead"

    # --- the tools themselves ------------------------------------------------

    def _handlers(self) -> dict[str, Callable[[dict[str, Any]], dict[str, Any]]]:
        return {
            "list_accounts": self._list_accounts,
            "list_bills": self._list_bills,
            "list_cards": self._list_cards,
            "get_balance": self._get_balance,
            "get_spending": self._get_spending,
            "transfer_money": self._transfer_money,
            "pay_bill": self._pay_bill,
            "freeze_card": self._freeze_card,
            "order_card": self._order_card,
        }

    def _list_accounts(self, _: dict[str, Any]) -> dict[str, Any]:
        return {
            "accounts": [
                {
                    "name": a.name,
                    "balance": format_money(a.balance_cents),
                    "balance_cents": a.balance_cents,
                }
                for a in self.bank.accounts()
            ]
        }

    def _list_bills(self, _: dict[str, Any]) -> dict[str, Any]:
        return {
            "bills": [
                {
                    "name": b.name,
                    "amount": format_money(b.amount_cents),
                    "amount_cents": b.amount_cents,
                    "paid": b.paid,
                }
                for b in self.bank.bills()
            ]
        }

    def _list_cards(self, _: dict[str, Any]) -> dict[str, Any]:
        return {
            "cards": [
                {"last4": c.last4, "account": c.account, "frozen": c.frozen}
                for c in self.bank.cards()
            ]
        }

    def _get_balance(self, arguments: dict[str, Any]) -> dict[str, Any]:
        account = _require_text(arguments.get("account"), "account")
        cents = self.bank.balance(account)
        return {
            "account": account.strip().lower(),
            "balance": format_money(cents),
            "balance_cents": cents,
        }

    def _get_spending(self, arguments: dict[str, Any]) -> dict[str, Any]:
        period = _require_text(arguments.get("period"), "period")
        merchant = arguments.get("merchant")
        cents = self.bank.spending(period, merchant if isinstance(merchant, str) else None)
        out = {"period": period, "total": format_money(cents), "total_cents": cents}
        if merchant:
            out["merchant"] = merchant
        return out

    def _transfer_money(self, arguments: dict[str, Any]) -> dict[str, Any]:
        remaining = self.bank.transfer(
            arguments["amount_cents"], arguments["payee"], arguments["from_account"]
        )
        return {
            "status": "done",
            "sent": format_money(arguments["amount_cents"]),
            "sent_cents": arguments["amount_cents"],
            "payee": arguments["payee"],
            "remaining_balance": format_money(remaining),
            "remaining_balance_cents": remaining,
        }

    def _pay_bill(self, arguments: dict[str, Any]) -> dict[str, Any]:
        paid = self.bank.pay_bill(arguments["bill"])
        return {
            "status": "done",
            "bill": arguments["bill"],
            "paid": format_money(paid),
            "paid_cents": paid,
        }

    def _freeze_card(self, arguments: dict[str, Any]) -> dict[str, Any]:
        card = self.bank.freeze_card(arguments["last4"])
        return {
            "status": "done",
            "last4": card.last4,
            "frozen": True,
            "reported_fraud": bool(arguments.get("report_fraud")),
        }

    def _order_card(self, arguments: dict[str, Any]) -> dict[str, Any]:
        card = self.bank.order_card(arguments["account"])
        return {
            "status": "done",
            "account": card.account,
            "last4": card.last4,
            "arrives_in_working_days": 5,
        }
