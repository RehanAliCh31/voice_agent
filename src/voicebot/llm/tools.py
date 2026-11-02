"""The fifteen things the model is allowed to do.

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

from voicebot.bank.base import (
    CARD_CONTROLS,
    DISPUTE_REASONS,
    HANDOFF_REASONS,
    PERIODS,
    PIN_PURPOSES,
)

# How many ledger rows a spoken answer can carry. A reply is read aloud, so a
# list long enough to be useful on a screen is unusable in a conversation.
DEFAULT_TRANSACTIONS = 5
MAX_TRANSACTIONS = 20

# Tools that move money or change a card. Each one is called twice: the first
# call returns a summary and a token and changes nothing, the second carries
# the token back and executes. Enforced in the executor, not in the prompt.
CONFIRMING: frozenset[str] = frozenset(
    {
        "dispute_transaction",
        "set_card_control",
        "transfer_money",
        "pay_bill",
        "freeze_card",
        "order_card",
        "move_money",
    }
)

# Of those, the ones that actually move money. A yes is not enough for these:
# the PIN for the account the money leaves is asked for after the read-back
# and before anything happens. Freezing a card or filing a dispute is not
# here -- both are reversible, and neither moves a penny.
#
# Enforced in the executor, like the gate above, so it holds whether the
# customer pressed the button or the model called the tool back with its
# token. A prompt that asks nicely is not a control.
PIN_GUARDED: frozenset[str] = frozenset(
    {
        "transfer_money",
        "pay_bill",
        "move_money",
    }
)

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
    _function(
        "list_transactions",
        "List movements on the account over a named period -- what was charged, "
        "by whom, and when. Ask the customer which period they mean before "
        "calling this; there is no default and a call without one is refused. "
        "Use this when the customer asks what a charge was or what they have "
        "paid someone; use get_spending when they want a total rather than the "
        "individual rows.",
        {
            "account": _ACCOUNT,
            "period": {
                "type": "string",
                "enum": list(PERIODS),
                "description": (
                    "Which period to list. Required. Ask the customer rather "
                    "than choosing one for them."
                ),
            },
            "merchant": {
                "type": "string",
                "description": "Optional. Only movements involving this merchant.",
            },
            "limit": {
                "type": "integer",
                "description": (
                    "How many to return, newest first. Defaults to "
                    f"{DEFAULT_TRANSACTIONS}, and never more than {MAX_TRANSACTIONS} "
                    "-- the reply is read aloud."
                ),
            },
        },
    ),
    # --- leaving ---
    _function(
        "hand_off",
        "Pass the conversation to a person, and tell the customer their "
        "reference. Call this as soon as one of the reasons applies -- "
        "especially fraud, a complaint, or money trouble. Do not try to "
        "settle those yourself, and never ask the customer to confirm that "
        "they want a human. Never call it to answer a question -- what you can "
        "do, how this works, anything general: answer those yourself.",
        {
            "reason": {
                "type": "string",
                "enum": list(HANDOFF_REASONS),
                "description": (
                    "Why a person is needed. 'not permitted' is for anything "
                    "beyond what your tools can do, such as waiving a fee."
                ),
            },
            "note": {
                "type": "string",
                "description": "One short line for the person picking it up.",
            },
        },
        ["reason"],
    ),
    # --- writes: each needs confirmation ---
    _function(
        "transfer_money",
        "Send money OUT of the bank to a payee -- a person, not an account. "
        "Returns needs_confirmation on the first call: read the summary back "
        "to the customer and wait for them to agree. To put money into the "
        "customer's own savings or checking, use move_money instead.",
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
        "dispute_transaction",
        "File a dispute against one charge the customer says is wrong. Find "
        "the charge with list_transactions first and pass its id. Returns "
        "needs_confirmation on the first call. This files a claim and nothing "
        "more: you cannot decide it, no money comes back now, and you must "
        "not tell the customer the charge will be refunded.",
        {
            "transaction_id": {
                "type": "integer",
                "description": "The id of the charge, as list_transactions reports it.",
            },
            "reason": {
                "type": "string",
                "enum": list(DISPUTE_REASONS),
                "description": "What the customer says is wrong with it.",
            },
            "note": {
                "type": "string",
                "description": "One short line in the customer's own words.",
            },
        },
        ["transaction_id", "reason"],
    ),
    _function(
        "move_money",
        "Move money between two of the customer's OWN accounts, such as "
        "checking to savings. Returns needs_confirmation on the first call. "
        "Use this, never transfer_money, when the destination is an account "
        "rather than a person -- transfer_money sends money out of the bank.",
        {
            "amount_cents": {
                "type": "integer",
                "description": (
                    "The amount in whole cents, US dollars. 5000 means $50.00. "
                    "Must be a positive whole number — never a decimal."
                ),
            },
            "from_account": _ACCOUNT,
            "to_account": _ACCOUNT,
        },
        ["amount_cents", "from_account", "to_account"],
    ),
    _function(
        "pay_bill",
        "Pay one bill in full. Returns needs_confirmation on the first call.",
        {"bill": {"type": "string", "description": "The bill name, as list_bills reports it."}},
        ["bill"],
    ),
    _function(
        "freeze_card",
        "Freeze a card so nothing further goes through on it, and optionally "
        "send a replacement. Returns needs_confirmation on the first call. "
        "Use this for a card reported lost or stolen: it is one event and one "
        "confirmation, so do not call order_card as well. Ask whether the "
        "card was lost or stolen -- stolen means report_fraud.",
        {
            "last4": {
                "type": "string",
                "description": "The card's last four digits, as list_cards reports them.",
            },
            "report_fraud": {
                "type": "boolean",
                "description": (
                    "True when the card was stolen or used by someone else, "
                    "rather than simply mislaid. This reaches a specialist."
                ),
            },
            "order_replacement": {
                "type": "boolean",
                "description": "True to send a new card on the same account.",
            },
        },
        ["last4"],
    ),
    _function(
        "ask_for_pin",
        "Put the PIN keypad on the customer's screen. Call this before "
        "showing card details and before changing a PIN -- those need the "
        "PIN and you cannot take one. Never ask the customer to say a PIN, "
        "read one back, or repeat one: tell them it is on screen.",
        {
            "last4": {
                "type": "string",
                "description": "The card's last four digits, as list_cards reports them.",
            },
            "purpose": {
                "type": "string",
                "enum": list(PIN_PURPOSES),
                "description": (
                    "'details' to show what is printed on the card, 'reset' " "to change the PIN."
                ),
            },
        },
        ["last4", "purpose"],
    ),
    _function(
        "set_card_control",
        "Allow or refuse one kind of spending on a card: online purchases, "
        "shop purchases, or cash machines. Returns needs_confirmation on the "
        "first call. Use freeze_card instead to stop the card entirely.",
        {
            "last4": {
                "type": "string",
                "description": "The card's last four digits, as list_cards reports them.",
            },
            "control": {
                "type": "string",
                "enum": list(CARD_CONTROLS),
                "description": (
                    "Which setting: 'ecommerce' for online, 'retail' for shops, "
                    "'atm' for cash machines."
                ),
            },
            "enabled": {
                "type": "boolean",
                "description": "True to allow it, false to refuse it.",
            },
        },
        ["last4", "control", "enabled"],
    ),
    _function(
        "order_card",
        "Order a replacement card for an account. Returns needs_confirmation on " "the first call.",
        {"account": _ACCOUNT},
        ["account"],
    ),
)
