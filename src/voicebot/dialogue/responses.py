"""Everything the bot can say.

Templates, not generation. A generative model asked about a balance will
eventually invent one, and an invented balance is worse than no bot. Templates
are also testable: every string the bot can produce is in this file.
"""

from __future__ import annotations

from voicebot.dialogue.mock_backend import format_money
from voicebot.dialogue.slots import SLOT_PROMPTS

# Kept deliberately short. Every word here is SPOKEN, and synthesis is the
# slowest stage in the pipeline, so a wordier refusal costs the user seconds
# of waiting for a sentence they did not want in the first place.
CAPABILITIES = "balances, transfers, bills, spending and cards"


# Small talk the bot answers directly rather than refusing. An assistant
# that cannot say hello feels broken, and each of these costs one short
# sentence. They act on nothing, so there is no risk in answering them.
SOCIAL: dict[str, str] = {
    "greeting": f"Hello. I can help with {CAPABILITIES}.",
    "goodbye": "Goodbye.",
    "thank_you": "You're welcome.",
    "what_can_i_ask_you": f"I handle {CAPABILITIES}.",
    "who_do_you_work_for": "I'm a demo banking assistant.",
    "what_is_your_name": "I'm a banking assistant. What can I do for you?",
}


def social(intent: str) -> str:
    """Answer small talk. Only called for an intent in SOCIAL."""
    return SOCIAL[intent]


def ask_for(slot: str) -> str:
    """Prompt for one missing slot."""
    return SLOT_PROMPTS.get(slot, f"What {slot.replace('_', ' ')} should I use?")


def confirm(intent: str, slots: dict[str, str], cents: int | None, currency: str) -> str:
    """Read the pending action back before doing it."""
    if intent == "transfer":
        amount = format_money(cents, currency) if cents is not None else slots.get("amount", "")
        account = slots.get("account_type", "checking")
        return f"Transfer {amount} to {slots.get('payee', '')} from {account}. Shall I confirm?"
    if intent == "pay_bill":
        return f"Pay your {slots.get('bill_type', '')} bill in full. Shall I confirm?"
    if intent == "freeze_account":
        return f"Freeze the card ending {slots.get('card_last4', '')}. Shall I confirm?"
    if intent == "report_fraud":
        return (
            f"Report the card ending {slots.get('card_last4', '')} as fraudulent "
            "and freeze it. Shall I confirm?"
        )
    if intent == "new_card":
        return f"Order a new card for {slots.get('account_type', '')}. Shall I confirm?"
    return "Shall I go ahead?"


def balance_answer(account: str, cents: int) -> str:
    return f"Your {account} balance is {format_money(cents)}."


def transfer_answer(cents: int, currency: str, payee: str, account: str, remaining: int) -> str:
    return (
        f"Done. {format_money(cents, currency)} sent to {payee}. "
        f"Your {account} balance is now {format_money(remaining)}."
    )


def bill_answer(bill_type: str, cents: int) -> str:
    return f"Paid your {bill_type} bill, {format_money(cents)}."


def spending_answer(period: str, cents: int, merchant: str | None) -> str:
    where = f" at {merchant}" if merchant else ""
    if cents == 0:
        return f"I don't see any spending{where} {period}."
    return f"You spent {format_money(cents)}{where} {period}."


def freeze_answer(last4: str) -> str:
    return f"The card ending {last4} is now frozen. Nothing further will go through on it."


def fraud_answer(last4: str) -> str:
    return (
        f"I've reported the card ending {last4} and frozen it. "
        "A replacement will arrive in five working days."
    )


def new_card_answer(account: str) -> str:
    return f"A new card for {account} is on its way. It should arrive in five working days."


def unsupported(intent: str) -> str:
    """Recognised, but outside what this bot acts on. Say so rather than bluff."""
    spoken = intent.replace("_", " ")
    return f"That sounded like '{spoken}'. I handle {CAPABILITIES}."


def out_of_scope() -> str:
    return f"I only handle banking: {CAPABILITIES}."


def clarify() -> str:
    return "Sorry, I didn't catch that. Could you say it again?"


def cancelled() -> str:
    return "No problem, I've cancelled that. Anything else?"


def give_up() -> str:
    return "Let's start over. What would you like to do?"


def failed(reason: str) -> str:
    """A request understood correctly that the backend refused."""
    return reason
