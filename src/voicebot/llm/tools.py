"""The nine things the model is allowed to do.

This file is the whole surface between a fluent model and a customer's money.
Anything not described here, the bot cannot do — which is the point. Adding a
tool is a deliberate act, reviewed like any other change to what the product
can do.

Descriptions are written for the model, not for us. They are the only place it
learns that amounts are integer cents, that the bank holds one currency, and
that a confirming tool has to be called twice.
"""

from __future__ import annotations

from typing import Any

from voicebot.bank.base import PERIODS

# Tools that move money or change a card. Each one is called twice: the first
# call returns a summary and a token and changes nothing, the second carries
# the token back and executes. Enforced in the executor, not in the prompt.
CONFIRMING: frozenset[str] = frozenset({"transfer_money", "pay_bill", "freeze_card", "order_card"})

_TOKEN = {
    "type": "string",
    "description": (
        "The confirmation_token from your previous call to this tool. "
        "Include it ONLY after the customer has agreed out loud. "
        "Omit it on the first call."
    ),
}


def _function(
    name: str,
    description: str,
    properties: dict[str, Any] | None = None,
    required: list[str] | None = None,
) -> dict[str, Any]:
    """One schema in OpenAI function format."""
    properties = dict(properties or {})
    if name in CONFIRMING:
        properties["confirmation_token"] = _TOKEN
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required or [],
            },
        },
    }


_ACCOUNT = {
    "type": "string",
    "description": "The account name, exactly as list_accounts reports it.",
}

TOOL_SCHEMAS: tuple[dict[str, Any], ...] = (
    # --- discovery: ask rather than guess ---
    _function(
        "list_accounts",
        "List the customer's accounts and their balances. Use this when you are "
        "not certain which accounts exist.",
    ),
    _function(
        "list_bills",
        "List the bills on file, their amounts, and whether each is already paid.",
    ),
    _function(
        "list_cards",
        "List the customer's cards by their last four digits. Use this before "
        "freezing a card rather than asking the customer to read the digits out.",
    ),
    # --- reads ---
    _function(
        "get_balance",
        "Look up the balance of one account.",
        {"account": _ACCOUNT},
        ["account"],
    ),
    _function(
        "get_spending",
        "Total the customer's spending over a named period, optionally for one "
        "merchant. Only the listed periods are understood.",
        {
            "period": {
                "type": "string",
                "enum": list(PERIODS),
                "description": "Which period to total.",
            },
            "merchant": {
                "type": "string",
                "description": "Optional. Limit the total to this merchant.",
            },
        },
        ["period"],
    ),
    # --- writes: each needs confirmation ---
    _function(
        "transfer_money",
        "Send money to a payee. Returns needs_confirmation on the first call: "
        "read the summary back to the customer and wait for them to agree.",
        {
            "amount_cents": {
                "type": "integer",
                "description": (
                    "The amount in whole cents, US dollars. 5000 means $50.00. "
                    "Must be a positive whole number — never a decimal."
                ),
            },
            "payee": {"type": "string", "description": "Who to send the money to."},
            "from_account": _ACCOUNT,
        },
        ["amount_cents", "payee", "from_account"],
    ),
    _function(
        "pay_bill",
        "Pay one bill in full. Returns needs_confirmation on the first call.",
        {"bill": {"type": "string", "description": "The bill name, as list_bills reports it."}},
        ["bill"],
    ),
    _function(
        "freeze_card",
        "Freeze a card so nothing further goes through on it. Returns "
        "needs_confirmation on the first call.",
        {
            "last4": {
                "type": "string",
                "description": "The card's last four digits, as list_cards reports them.",
            },
            "report_fraud": {
                "type": "boolean",
                "description": "True if the customer is reporting fraud rather than "
                "just freezing the card.",
            },
        },
        ["last4"],
    ),
    _function(
        "order_card",
        "Order a replacement card for an account. Returns needs_confirmation on " "the first call.",
        {"account": _ACCOUNT},
        ["account"],
    ),
)
