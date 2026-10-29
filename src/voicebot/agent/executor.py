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
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from voicebot.bank.base import (
    BILL_ACCOUNT,
    CARD_CONTROLS,
    DISPUTE_REASONS,
    MODEL_PIN_PURPOSES,
    PERIODS,
    Bank,
    BankError,
    StorageError,
    Transaction,
    account_not_payee_message,
    format_money,
    unknown_control_message,
    unknown_dispute_reason_message,
    unknown_payee_message,
    unknown_purpose_message,
)
from voicebot.errors import DatabaseError, ToolError
from voicebot.llm.base import ToolCall
from voicebot.llm.prompts import PIN_NEEDED
from voicebot.llm.tools import (
    CONFIRMING,
    DEFAULT_TRANSACTIONS,
    MAX_TRANSACTIONS,
    PIN_GUARDED,
)
from voicebot.utils.logging import get_logger
from voicebot.utils.observability import report

log = get_logger("agent.executor")

# How long a read-back summary stays valid. Long enough for a person to think
# about it, short enough that a forgotten confirmation cannot be completed by
# a later, unrelated turn.
DEFAULT_TTL_S = 300.0


@dataclass(frozen=True)
class ToolOutcome:
    """What a tool produced, and which of its figures the bank actually gave.

    `sourced_cents` is the whole of what the numeric guard will trust. It is
    reported explicitly rather than parsed back out of `result`, because
    `result` contains strings the model supplied -- an echoed account name, a
    payee, an error quoting the bad argument -- and a guard that reads those
    can be fed any figure the model likes.
    """

    result: dict[str, Any]
    sourced_cents: tuple[int, ...] = ()
    # Set when the tool itself failed, as the message the loop then says --
    # never the model's to put into its own words, because a model told
    # "the database is down" goes on to guess. TOOL_FAILED: nothing was done.
    # OUTCOME_UNKNOWN: a write was under way and may or may not have landed.
    failed: str | None = None


TOOL_FAILED = "tool_failed"
OUTCOME_UNKNOWN = "outcome_unknown"


@dataclass
class PendingAction:
    """An action that has been read back to the customer but not performed."""

    tool: str
    arguments: dict[str, Any]
    token: str
    summary: str
    created_at: float
    # Set once the API has checked a PIN against the bank for this exact
    # action. It lives here, and not on the executor, so that agreeing to a
    # different amount cannot inherit the PIN given for the old one: changing
    # the arguments builds a fresh PendingAction, and this starts false again.
    pin_verified: bool = False

    def expired(self, ttl_s: float, now: float) -> bool:
        # >= so that a ttl of zero means "no validity at all" rather than
        # "valid until the clock happens to tick".
        return now - self.created_at >= ttl_s


def _error(message: str, hint: str) -> ToolOutcome:
    # No sourced figures: an error message may quote the model's own
    # argument back, and nothing in it came from the database.
    return ToolOutcome({"error": message, "hint": hint})


def _require_flag(value: Any, field: str, default: bool = False) -> bool:
    """A yes or no, and nothing that merely looks like one.

    `bool("false")` is True. A model that sends the word rather than the
    literal would otherwise report fraud, order a card or switch a control on
    when it meant the opposite.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "yes"):
        return True
    if isinstance(value, str) and value.strip().lower() in ("false", "no"):
        return False
    raise ValueError(f"{field} must be true or false.")


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


def _optional_text(value: Any) -> str | None:
    """A filter the caller may omit. Only a real string narrows the query.

    An empty or non-string value means "no filter" rather than a filter on
    nothing, which would quietly match no rows and answer a question the
    customer did not ask.
    """
    return value.strip() if isinstance(value, str) and value.strip() else None


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string.")
    return value.strip()


class ToolExecutor:
    """Runs tools for one conversation, and holds its one pending action."""

    def __init__(
        self,
        bank: Bank,
        ttl_s: float = DEFAULT_TTL_S,
        session_id: str = "",
    ) -> None:
        self.bank = bank
        self.ttl_s = ttl_s
        # Which conversation this executor belongs to, and how far into it we
        # are. A handoff has to say where it was raised so the person picking
        # it up can read the exchange that caused it. The agent keeps the turn
        # in step; the executor never advances it itself.
        self.session_id = session_id
        self.turn_index = 0
        # The case this conversation was handed to a person under, once it
        # has been. Asking again returns the same one.
        self.handed_off: tuple[str, str] | None = None
        self.pending: PendingAction | None = None
        # Counts balance-changing writes. A figure reported before a write is
        # stale after it, and the count is how a caller knows. Kept here
        # rather than in the agent so it cannot be bypassed -- the Yes button
        # reaches the executor directly.
        self.writes = 0

    # --- the public surface --------------------------------------------------

    def execute(self, tool_call: ToolCall) -> ToolOutcome:
        """Run one tool call, or ask for confirmation first."""
        handler = self._handlers().get(tool_call.name)
        if handler is None:
            return _error(
                f"There is no tool called {tool_call.name!r}.",
                f"Available tools: {', '.join(sorted(self._handlers()))}.",
            )

        arguments = {k: v for k, v in tool_call.arguments.items() if k != "confirmation_token"}
        token = tool_call.arguments.get("confirmation_token")

        # A write is under way when a gated tool comes back with its token,
        # or for the one ungated tool that writes. A failure part-way through
        # one of those may or may not have landed, and saying "nothing has
        # changed" then could make a customer pay twice.
        writing = tool_call.name == "hand_off" or (tool_call.name in CONFIRMING and bool(token))
        try:
            if tool_call.name in CONFIRMING:
                return self._gated(tool_call.name, arguments, token, handler)
            return handler(arguments)
        except BankError as exc:
            # The bank writes its refusals to be read aloud, so the message is
            # the hint: it says both what went wrong and what is allowed.
            return _error(str(exc), str(exc))
        except (ValueError, KeyError) as exc:
            return _error(str(exc), "Check the tool's parameters and try again.")
        except (sqlite3.Error, StorageError) as exc:
            return self._storage_failed(tool_call.name, exc, writing)
        except Exception as exc:  # a bug in a tool must not leave the turn half-built
            return self._crashed(tool_call.name, exc, writing)

    def confirm(self) -> ToolOutcome:
        """Perform the pending action. Called by the Yes button, not the model."""
        pending = self.pending
        if pending is None:
            return _error(
                "There's nothing waiting to be confirmed.",
                "No action was pending, so nothing happened.",
            )
        if pending.expired(self.ttl_s, time.monotonic()):
            # Sessions live for 30 minutes and the buttons stay on screen, so
            # without this a customer could return from lunch and confirm a
            # summary they no longer remember.
            self.pending = None
            log.warning("confirmation expired before the button was pressed: %s", pending.tool)
            return _error(
                "That confirmation has expired.",
                f"More than {int(self.ttl_s)}s passed. Nothing has changed; please ask again.",
            )
        if pending.tool in PIN_GUARDED and not pending.pin_verified:
            # Yes was not enough. The action stays pending while the keypad is
            # on screen, so cancelling the PIN leaves the customer exactly
            # where they were rather than losing what they agreed to.
            try:
                return self._needs_pin(pending)
            except (sqlite3.Error, StorageError) as exc:
                # Which card to ask about could not be read; nothing ran, and
                # the action is still there to approve once it can be.
                return self._storage_failed(pending.tool, exc, writing=False)
        self.pending = None
        try:
            return self._handlers()[pending.tool](pending.arguments)
        except BankError as exc:
            return _error(str(exc), str(exc))
        except (ValueError, KeyError) as exc:
            return _error(str(exc), "The pending action could not be completed.")
        except (sqlite3.Error, StorageError) as exc:
            return self._storage_failed(pending.tool, exc, writing=True)
        except Exception as exc:  # a bug in a tool must not leave the turn half-built
            return self._crashed(pending.tool, exc, writing=True)

    def _storage_failed(self, tool: str, exc: Exception, writing: bool) -> ToolOutcome:
        """The database failed under a tool: a refusal for a read, uncertainty for a write."""
        report(DatabaseError(f"{tool}: {type(exc).__name__}", component="bank", cause=exc))
        return self._failure(writing)

    def _crashed(self, tool: str, exc: Exception, writing: bool) -> ToolOutcome:
        log.error("%s raised %s", tool, type(exc).__name__, exc_info=exc)
        report(ToolError(f"{tool} raised {type(exc).__name__}", component=tool, cause=exc))
        return self._failure(writing)

    def _failure(self, writing: bool) -> ToolOutcome:
        if writing:
            # It may have happened. Every figure read before it is treated as
            # stale, the same as after a write that is known to have landed.
            self.writes += 1
            return ToolOutcome(
                {
                    "error": "It could not be confirmed whether that went through.",
                    "hint": "Do not repeat it. The customer should check their balance first.",
                },
                failed=OUTCOME_UNKNOWN,
            )
        return ToolOutcome(
            {
                "error": "That could not be looked up just now.",
                "hint": "Nothing was changed. Do not guess the answer.",
            },
            failed=TOOL_FAILED,
        )

    def pin_accepted(self) -> None:
        """Record that a PIN was given and the bank accepted it.

        Called by the API, which does the checking. There is no parameter for
        a PIN and nothing here ever holds one: what the customer types goes
        from the keypad to the bank, never through the model, the transcript
        or the audit log. All that crosses this line is the fact that it was
        right, and it is pinned to the one action it was asked for.
        """
        if self.pending is not None:
            self.pending.pin_verified = True

    def pin_card(self) -> str | None:
        """Which card's PIN authorises the pending action, if one is pending.

        The card on the account the money leaves. The API needs this to know
        what to check the PIN against, and asking the executor keeps that
        decision in one place rather than in both.
        """
        pending = self.pending
        if pending is None or pending.tool not in PIN_GUARDED:
            return None
        return self._card_on(self._account_charged(pending))

    def cancel(self) -> ToolOutcome:
        """Discard the pending action. Called by the Cancel button."""
        pending = self.pending
        self.pending = None
        if pending is None:
            return ToolOutcome({"status": "cancelled", "detail": "There was nothing pending."})
        return ToolOutcome({"status": "cancelled", "detail": f"Cancelled: {pending.summary}"})

    # --- the gate ------------------------------------------------------------

    def _gated(
        self,
        name: str,
        arguments: dict[str, Any],
        token: Any,
        handler: Callable[[dict[str, Any]], ToolOutcome],
    ) -> ToolOutcome:
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
            if name in PIN_GUARDED and not pending.pin_verified:
                # The same gate the button meets. Without it the model could
                # carry its own token back and move money with no PIN at all,
                # which would make the whole thing a suggestion.
                return self._needs_pin(pending)
            self.pending = None
            return handler(validated)

        return self._begin(name, validated)

    def _needs_pin(self, pending: PendingAction) -> ToolOutcome:
        """Hold the agreed action and ask for the PIN. Nothing has moved yet."""
        last4 = self._card_on(self._account_charged(pending))
        if last4 is None:
            # No card on that account means no PIN can be taken for it, and a
            # PIN that cannot be taken must not become a PIN that is skipped.
            # The action is dropped rather than quietly let through.
            self.pending = None
            log.warning("no card on the account %s would debit; refusing", pending.tool)
            return _error(
                "I can't take a PIN for that account, so I can't move the money.",
                "No card belongs to the account the money would leave. Nothing happened.",
            )
        log.info("holding %s for a PIN on card %s", pending.tool, last4)
        return ToolOutcome(
            {
                "status": "pin_required",
                "last4": last4,
                "purpose": "payment",
                "summary": pending.summary,
                "detail": PIN_NEEDED,
            }
        )

    @staticmethod
    def _account_charged(pending: PendingAction) -> str:
        """The account the money comes out of."""
        if pending.tool == "pay_bill":
            return BILL_ACCOUNT
        return str(pending.arguments.get("from_account", "")).strip().lower()

    def _card_on(self, account: str) -> str | None:
        """The card belonging to an account, by its last four digits."""
        for card in self.bank.cards():
            if card.account == account:
                return card.last4
        return None

    def _begin(self, name: str, arguments: dict[str, Any]) -> ToolOutcome:
        """Record what we are about to do, and say it back. Nothing happens yet."""
        summary = self._summarise(name, arguments)
        self.pending = PendingAction(
            tool=name,
            arguments=arguments,
            token=secrets.token_urlsafe(8),
            summary=summary,
            created_at=time.monotonic(),
        )
        # The proposed amount is reportable so the bot can read the summary
        # back. It is the one figure in the sourced set the model chose, and
        # the customer is looking at a confirmation prompt for exactly it.
        proposed = arguments.get("amount_cents")
        if not isinstance(proposed, int):
            proposed = self._bill_amount(arguments["bill"]) if name == "pay_bill" else None
        return ToolOutcome(
            {"status": "needs_confirmation", "token": self.pending.token, "summary": summary},
            sourced_cents=(proposed,) if isinstance(proposed, int) else (),
        )

    def _bill_amount(self, bill: str) -> int | None:
        """What a bill comes to, or None if there is no such bill."""
        found = next((b for b in self.bank.bills() if b.name == bill.lower()), None)
        return found.amount_cents if found else None

    def _validate(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Check a confirming tool's arguments. Raises ValueError with a hint."""
        if name == "transfer_money":
            return {
                "amount_cents": _require_positive_int(
                    arguments.get("amount_cents"), "amount_cents"
                ),
                "payee": self._known_payee(_require_text(arguments.get("payee"), "payee")),
                "from_account": _require_text(arguments.get("from_account"), "from_account"),
            }
        if name == "pay_bill":
            return {"bill": _require_text(arguments.get("bill"), "bill")}
        if name == "freeze_card":
            last4 = _require_text(arguments.get("last4"), "last4")
            if last4 not in {card.last4 for card in self.bank.cards()}:
                known = ", ".join(card.last4 for card in self.bank.cards())
                raise ValueError(f"I don't have a card ending {last4}. I have: {known}.")
            return {
                "last4": last4,
                "report_fraud": _require_flag(arguments.get("report_fraud"), "report_fraud"),
                "order_replacement": _require_flag(
                    arguments.get("order_replacement"), "order_replacement"
                ),
            }
        if name == "order_card":
            return {"account": _require_text(arguments.get("account"), "account")}
        if name == "set_card_control":
            control = _require_text(arguments.get("control"), "control").lower()
            if control not in CARD_CONTROLS:
                raise ValueError(unknown_control_message(control))
            last4 = _require_text(arguments.get("last4"), "last4")
            if last4 not in {card.last4 for card in self.bank.cards()}:
                # Refused before the read-back: agreeing to change a setting
                # on a card that is not there settles nothing.
                known = ", ".join(card.last4 for card in self.bank.cards())
                raise ValueError(f"I don't have a card ending {last4}. I have: {known}.")
            return {
                "last4": last4,
                "control": control,
                "enabled": _require_flag(arguments.get("enabled"), "enabled"),
            }
        if name == "dispute_transaction":
            charge = self._charge(arguments.get("transaction_id"))
            reason = _require_text(arguments.get("reason"), "reason").lower()
            if reason not in DISPUTE_REASONS:
                raise ValueError(unknown_dispute_reason_message(reason))
            raised = {case.transaction_id for case in self.bank.disputes()}
            if charge.id in raised:
                # Checked here as well as in the bank: confirming a dispute
                # that will be rejected means reading it back as though it
                # will be filed, and the customer agreeing to nothing.
                raise ValueError(
                    f"The {charge.merchant} charge for "
                    f"{format_money(abs(charge.cents))} has already been disputed."
                )
            return {
                "transaction_id": charge.id,
                "reason": reason,
                "note": _optional_text(arguments.get("note")),
            }
        if name == "move_money":
            source = _require_text(arguments.get("from_account"), "from_account")
            target = _require_text(arguments.get("to_account"), "to_account")
            if source.lower() == target.lower():
                # Checked here, not only in the bank: validation is what keeps
                # a nonsense request from becoming a pending action. Without
                # it the customer is read back "move $500 from checking to
                # checking" and asked to agree to it.
                raise ValueError(f"{source} and {target} are the same account.")
            return {
                "amount_cents": _require_positive_int(
                    arguments.get("amount_cents"), "amount_cents"
                ),
                "from_account": source,
                "to_account": target,
            }
        raise KeyError(f"{name} is not a confirming tool")

    def _known_payee(self, name: str) -> str:
        """Check who is being paid, before anything is read back.

        The bank refuses an unknown payee too, and must: it is reachable from
        the command line. But refusing there happens after the customer has
        already agreed to pay someone who does not exist.
        """
        key = name.strip().lower()
        if key in {payee.name for payee in self.bank.payees()}:
            return key
        if key in {account.name for account in self.bank.accounts()}:
            raise ValueError(account_not_payee_message(key))
        raise ValueError(unknown_payee_message(name, tuple(p.name for p in self.bank.payees())))

    def _summarise(self, name: str, arguments: dict[str, Any]) -> str:
        """The sentence the customer agrees to. Built from values, never generated."""
        if name == "transfer_money":
            amount = format_money(arguments["amount_cents"])
            return f"Transfer {amount} to {arguments['payee']} from {arguments['from_account']}"
        if name == "pay_bill":
            # The amount belongs in the sentence the customer agrees to. It
            # used to be absent, so the figure they heard came from an earlier
            # lookup rather than from the thing about to happen.
            due = self._bill_amount(arguments["bill"])
            if due is not None:
                return f"Pay your {arguments['bill']} bill in full, {format_money(due)}"
            return f"Pay your {arguments['bill']} bill in full"
        if name == "freeze_card":
            # Built from the parts rather than written out four times, so
            # every combination reads as one sentence about one event.
            card = f"the card ending {arguments['last4']}"
            opening = (
                f"Report {card} as stolen and freeze it"
                if arguments.get("report_fraud")
                else f"Freeze {card}"
            )
            if arguments.get("order_replacement"):
                return f"{opening}, and send a replacement"
            return opening
        if name == "order_card":
            return f"Order a new card for {arguments['account']}"
        if name == "set_card_control":
            named = {
                "ecommerce": "online purchases",
                "retail": "shop purchases",
                "atm": "cash machines",
            }
            state = "on" if arguments["enabled"] else "off"
            return (
                f"Turn {state} {arguments['control']} "
                f"({named[arguments['control']]}) on the card ending {arguments['last4']}"
            )
        if name == "dispute_transaction":
            charge = self.bank.transaction(arguments["transaction_id"])
            if charge is None:  # pragma: no cover - validated before this runs
                return "Raise a dispute"
            return (
                f"Dispute the {format_money(abs(charge.cents))} {charge.merchant} "
                f"charge as {arguments['reason']}"
            )
        if name == "move_money":
            amount = format_money(arguments["amount_cents"])
            return (
                f"Move {amount} from {arguments['from_account']} " f"to {arguments['to_account']}"
            )
        return "Go ahead"

    # --- the tools themselves ------------------------------------------------

    def _handlers(self) -> dict[str, Callable[[dict[str, Any]], ToolOutcome]]:
        return {
            "list_accounts": self._list_accounts,
            "list_bills": self._list_bills,
            "list_cards": self._list_cards,
            "get_balance": self._get_balance,
            "get_spending": self._get_spending,
            "list_transactions": self._list_transactions,
            "transfer_money": self._transfer_money,
            "pay_bill": self._pay_bill,
            "freeze_card": self._freeze_card,
            "order_card": self._order_card,
            "move_money": self._move_money,
            "hand_off": self._hand_off,
            "dispute_transaction": self._dispute_transaction,
            "set_card_control": self._set_card_control,
            "ask_for_pin": self._ask_for_pin,
        }

    def _list_accounts(self, _: dict[str, Any]) -> ToolOutcome:
        accounts = self.bank.accounts()
        return ToolOutcome(
            {
                "accounts": [
                    {
                        "name": a.name,
                        "balance": format_money(a.balance_cents),
                        "balance_cents": a.balance_cents,
                    }
                    for a in accounts
                ]
            },
            sourced_cents=tuple(a.balance_cents for a in accounts),
        )

    def _list_bills(self, _: dict[str, Any]) -> ToolOutcome:
        bills = self.bank.bills()
        return ToolOutcome(
            {
                "bills": [
                    {
                        "name": b.name,
                        "amount": format_money(b.amount_cents),
                        "amount_cents": b.amount_cents,
                        "paid": b.paid,
                    }
                    for b in bills
                ]
            },
            sourced_cents=tuple(b.amount_cents for b in bills),
        )

    def _list_cards(self, _: dict[str, Any]) -> ToolOutcome:
        return ToolOutcome(
            {
                "cards": [
                    {
                        "last4": c.last4,
                        "account": c.account,
                        "frozen": c.frozen,
                        "kind": c.kind,
                        "holder": c.holder,
                        "controls": c.controls,
                    }
                    for c in self.bank.cards()
                ]
            }
        )

    def _get_balance(self, arguments: dict[str, Any]) -> ToolOutcome:
        account = _require_text(arguments.get("account"), "account")
        cents = self.bank.balance(account)
        return ToolOutcome(
            {
                "account": account.strip().lower(),
                "balance": format_money(cents),
                "balance_cents": cents,
            },
            sourced_cents=(cents,),
        )

    def _get_spending(self, arguments: dict[str, Any]) -> ToolOutcome:
        period = _require_text(arguments.get("period"), "period")
        raw_merchant = arguments.get("merchant")
        # Only a string narrows the query, so only a string may label the
        # answer. Echoing a merchant that was not filtered on would put a
        # confident label on the wrong figure.
        merchant = raw_merchant if isinstance(raw_merchant, str) and raw_merchant.strip() else None
        cents = self.bank.spending(period, merchant)
        out: dict[str, Any] = {
            "period": period,
            "total": format_money(cents),
            "total_cents": cents,
        }
        if merchant:
            out["merchant"] = merchant
        return ToolOutcome(out, sourced_cents=(cents,))

    def _list_transactions(self, arguments: dict[str, Any]) -> ToolOutcome:
        # No default period on purpose. "The newest five" spans whatever
        # months happen to be nearest, so it listed last month's charges
        # beside this month's and called them all recent -- next to a screen
        # headed with one month. The customer decides the stretch of time.
        period = _optional_text(arguments.get("period"))
        if period is None:
            return _error(
                "I need to know which period you mean before I can list those.",
                "No period was given. Ask the customer which one they want -- "
                f"{', '.join(PERIODS)} -- and call this again with it.",
            )

        rows = self.bank.transactions(
            limit=self._row_count(arguments.get("limit")),
            account=_optional_text(arguments.get("account")),
            merchant=_optional_text(arguments.get("merchant")),
            period=period,
        )
        return ToolOutcome(
            {
                # Named back so the reply can say which stretch of time it
                # covers, and so an empty list reads as "nothing this week"
                # rather than as "nothing, ever".
                "period": period,
                "transactions": [
                    {
                        # The handle a dispute refers to. Two $10.99 charges
                        # are not interchangeable, and a merchant and amount
                        # cannot tell them apart.
                        "id": row.id,
                        "merchant": row.merchant,
                        "account": row.account,
                        "amount": format_money(abs(row.cents)),
                        "amount_cents": abs(row.cents),
                        "direction": "out" if row.spent else "in",
                        "occurred_at": row.occurred_at,
                    }
                    for row in rows
                ],
            },
            # Absolute, because that is what gets spoken. The ledger keeps the
            # sign and the reply states the direction in words; a guard
            # comparing against a negative would suppress every figure here.
            sourced_cents=tuple(abs(row.cents) for row in rows),
        )

    @staticmethod
    def _row_count(value: Any) -> int:
        """Clamp rather than refuse: an over-long list is not a bad request."""
        if isinstance(value, bool) or not isinstance(value, int):
            return DEFAULT_TRANSACTIONS
        return max(1, min(value, MAX_TRANSACTIONS))

    def _transfer_money(self, arguments: dict[str, Any]) -> ToolOutcome:
        sent = arguments["amount_cents"]
        remaining = self.bank.transfer(sent, arguments["payee"], arguments["from_account"])
        self.writes += 1
        return ToolOutcome(
            {
                "status": "done",
                "sent": format_money(sent),
                "sent_cents": sent,
                "payee": arguments["payee"],
                "remaining_balance": format_money(remaining),
                "remaining_balance_cents": remaining,
            },
            sourced_cents=(sent, remaining),
        )

    def _pay_bill(self, arguments: dict[str, Any]) -> ToolOutcome:
        paid = self.bank.pay_bill(arguments["bill"])
        self.writes += 1
        return ToolOutcome(
            {
                "status": "done",
                "bill": arguments["bill"],
                "paid": format_money(paid),
                "paid_cents": paid,
            },
            sourced_cents=(paid,),
        )

    def _charge(self, transaction_id: Any) -> Transaction:
        """The ledger row a dispute is about, or a sentence saying it is not there."""
        if isinstance(transaction_id, bool) or not isinstance(transaction_id, int):
            raise ValueError("transaction_id must be the id list_transactions reported.")
        charge = self.bank.transaction(transaction_id)
        if charge is None:
            raise ValueError("I can't find that charge on your account.")
        return charge

    def _dispute_transaction(self, arguments: dict[str, Any]) -> ToolOutcome:
        case = self.bank.open_dispute(
            arguments["transaction_id"], arguments["reason"], arguments["note"]
        )
        # No write counted: a dispute changes no balance. Saying otherwise
        # would mark every figure looked up before it as stale, for a tool
        # that moved nothing.
        return ToolOutcome(
            {
                "status": "filed",
                "reference": case.reference,
                "merchant": case.merchant,
                "amount": format_money(case.amount_cents),
                "amount_cents": case.amount_cents,
                "reason": case.reason,
                # Named `outcome` rather than `status` so the model cannot
                # read "filed" as a decision. It is open, and it stays open
                # until a person closes it.
                "outcome": case.status,
            },
            sourced_cents=(case.amount_cents,),
        )

    def _hand_off(self, arguments: dict[str, Any]) -> ToolOutcome:
        reason = _require_text(arguments.get("reason"), "reason")
        note = _optional_text(arguments.get("note"))
        if self.handed_off is not None:
            # Already with a colleague. Asked again -- by the model after an
            # error, by the rule and the model both -- it is the same case,
            # not a second one for someone to work twice.
            reference, opened_for = self.handed_off
            return ToolOutcome(
                {"status": "handed_off", "reference": reference, "reason": opened_for}
            )
        case = self.bank.open_handoff(self.session_id, self.turn_index, reason, note)
        self.handed_off = (case.reference, case.reason)

        # Whatever was waiting on a yes is dropped. A conversation that needs
        # a person must not also leave money one word away from moving, and a
        # customer reporting fraud is the likeliest of all to say "yes" to
        # something they have stopped listening to.
        if self.pending is not None:
            log.info("cancelling %s: the conversation is going to a person", self.pending.tool)
            self.pending = None

        return ToolOutcome(
            {
                "status": "handed_off",
                "reference": case.reference,
                "reason": case.reason,
            }
        )

    def _move_money(self, arguments: dict[str, Any]) -> ToolOutcome:
        moved = arguments["amount_cents"]
        source, target = arguments["from_account"], arguments["to_account"]
        from_balance, to_balance = self.bank.move(moved, source, target)
        self.writes += 1
        return ToolOutcome(
            {
                "status": "done",
                "moved": format_money(moved),
                "moved_cents": moved,
                "from_account": source,
                "from_balance": format_money(from_balance),
                "from_balance_cents": from_balance,
                "to_account": target,
                "to_balance": format_money(to_balance),
                "to_balance_cents": to_balance,
            },
            sourced_cents=(moved, from_balance, to_balance),
        )

    def _ask_for_pin(self, arguments: dict[str, Any]) -> ToolOutcome:
        """Ask the page for a PIN. Nothing here ever sees one.

        There is no parameter for a PIN and no way to pass one: what the
        customer types goes from the keypad to be checked, and never through
        the model, the transcript or the audit log.
        """
        last4 = _require_text(arguments.get("last4"), "last4")
        if last4 not in {card.last4 for card in self.bank.cards()}:
            known = ", ".join(card.last4 for card in self.bank.cards())
            raise ValueError(f"I don't have a card ending {last4}. I have: {known}.")

        purpose = _require_text(arguments.get("purpose"), "purpose").lower()
        if purpose not in MODEL_PIN_PURPOSES:
            raise ValueError(unknown_purpose_message(purpose))

        return ToolOutcome({"status": "pin_requested", "last4": last4, "purpose": purpose})

    def _set_card_control(self, arguments: dict[str, Any]) -> ToolOutcome:
        card = self.bank.set_card_control(
            arguments["last4"], arguments["control"], arguments["enabled"]
        )
        # No write counted: this changes what a card may do, not what an
        # account holds, so no figure looked up earlier has gone stale.
        return ToolOutcome(
            {
                "status": "done",
                "last4": card.last4,
                "control": arguments["control"],
                "enabled": arguments["enabled"],
                "controls": card.controls,
            }
        )

    def _freeze_card(self, arguments: dict[str, Any]) -> ToolOutcome:
        card = self.bank.freeze_card(arguments["last4"])
        stolen = bool(arguments.get("report_fraud"))

        replacement = None
        if arguments.get("order_replacement"):
            # Against the card that was lost, not against an account chosen
            # separately: the replacement draws on whatever the old one did.
            issued = self.bank.order_card(card.account)
            replacement = {
                "last4": issued.last4,
                "account": issued.account,
                "arrives_in_working_days": 5,
            }

        # A theft is a fraud report, and a fraud report reaches a person.
        # Opened here rather than left to the model to think of, for the same
        # reason the customer's own request for a human is.
        reference = None
        if stolen:
            case = self.bank.open_handoff(
                self.session_id,
                self.turn_index,
                "fraud",
                f"card ending {card.last4} reported stolen",
            )
            reference = case.reference
            log.info("card %s reported stolen; opened %s", card.last4, reference)

        return ToolOutcome(
            {
                "status": "done",
                "last4": card.last4,
                "frozen": True,
                "reported_fraud": stolen,
                "replacement": replacement,
                "reference": reference,
            }
        )

    def _order_card(self, arguments: dict[str, Any]) -> ToolOutcome:
        card = self.bank.order_card(arguments["account"])
        return ToolOutcome(
            {
                "status": "done",
                "account": card.account,
                "last4": card.last4,
                "arrives_in_working_days": 5,
            }
        )
