"""The dialogue manager: decides what the bot does with an understood utterance.

Order of checks matters and is not arbitrary:

1. A pending confirmation is resolved FIRST. While the bot is waiting on a
   yes or no, "yes" is an answer to its question, not a new request -- and the
   NLU, asked to classify the bare word "yes", will usually return something
   irrelevant with low confidence. Checking confirmation first stops the gate
   from throwing away the user's agreement.
2. Out-of-scope, then low confidence. Both decline rather than guess.
3. Only then is the utterance treated as a request.
"""

from __future__ import annotations

from voicebot.dialogue import responses
from voicebot.dialogue.mock_backend import BankError, MockBank
from voicebot.dialogue.slots import (
    INTENT_SPECS,
    is_affirmative,
    is_negative,
    parse_amount,
    parse_card_last4,
)
from voicebot.dialogue.states import Phase, phase_of, reset
from voicebot.schemas import ActionType, BotAction, DialogueState, Turn, Understanding
from voicebot.utils.logging import get_logger

log = get_logger("dialogue.manager")


class DialogueManager:
    """Drives one conversation. Hold one per session."""

    def __init__(
        self,
        bank: MockBank | None = None,
        clarify_threshold: float = 0.5,
        max_slot_attempts: int = 3,
    ) -> None:
        self.bank = bank or MockBank()
        self.clarify_threshold = clarify_threshold
        self.max_slot_attempts = max_slot_attempts

    # --- public API ----------------------------------------------------------

    def handle(self, text: str, understanding: Understanding, state: DialogueState) -> BotAction:
        """Advance the conversation by one turn and say what the bot does."""
        state.turn_index += 1
        action = self._decide(text, understanding, state)
        state.history.append(Turn(user_text=text, bot_text=action.text, intent=action.intent))
        log.info(
            "turn=%d phase=%s action=%s intent=%s",
            state.turn_index,
            phase_of(state).value,
            action.type.value,
            action.intent,
        )
        return action

    # --- decision logic ------------------------------------------------------

    def _decide(self, text: str, understanding: Understanding, state: DialogueState) -> BotAction:
        if phase_of(state) is Phase.CONFIRMING:
            return self._resolve_confirmation(text, understanding, state)

        # While a slot question is outstanding the reply is an ANSWER to it.
        # The NLU, asked to classify a fragment like "savings account", returns
        # something with low confidence that the gate below then rejects --
        # throwing away the very answer the bot asked for. Same reasoning as
        # the confirmation check above, one step earlier in the task.
        if phase_of(state) is Phase.COLLECTING:
            answered = self._answer_slot_question(text, understanding, state)
            if answered is not None:
                return answered

        if understanding.is_out_of_scope:
            reset(state)
            return BotAction(type=ActionType.OUT_OF_SCOPE, text=responses.out_of_scope())

        if understanding.intent_confidence < self.clarify_threshold:
            return BotAction(type=ActionType.CLARIFY, text=responses.clarify())

        intent = understanding.intent
        if intent in responses.SOCIAL:
            # Deliberately does not reset: saying hello mid-task should not
            # throw the task away.
            return BotAction(
                type=ActionType.ANSWER, text=responses.social(intent), intent=intent
            )

        if intent not in INTENT_SPECS:
            reset(state)
            return BotAction(
                type=ActionType.ANSWER, text=responses.unsupported(intent), intent=intent
            )

        # A new actionable intent replaces whatever was in progress. Users
        # change their minds mid-task and expect the bot to follow.
        if state.active_intent != intent:
            reset(state)
            state.active_intent = intent

        added = self._merge_slots(understanding, state)
        if added:
            state.slot_attempts = 0

        return self._advance(state)

    def _merge_slots(self, understanding: Understanding, state: DialogueState) -> int:
        """Copy newly extracted slots into the state. Returns how many were new."""
        added = 0
        for slot in understanding.slots:
            if not self._usable(slot.name, slot.value):
                continue
            if slot.name not in state.filled_slots:
                added += 1
            state.filled_slots[slot.name] = slot.value
        return added

    def _usable(self, name: str, value: str) -> bool:
        """Reject a slot value the backend could never act on.

        The slot head labels tokens in every utterance, including ones with no
        slots in them, so "what is my account balance" yields
        account_type="account". Accepting that made the bot answer "I don't
        recognise the account 'account'" when the honest reply is to ask which
        account. Dropping it here leaves the slot missing, and the existing
        prompt takes over.
        """
        if name == "account_type":
            return self.bank.knows_account(value)
        return True

    def _answer_slot_question(
        self, text: str, understanding: Understanding, state: DialogueState
    ) -> BotAction | None:
        """Read the utterance as the answer to the slot the bot just asked for.

        Returns None when it cannot be read that way, which lets the normal
        path treat it as a new request instead.
        """
        intent = state.active_intent
        if intent is None or intent not in INTENT_SPECS:
            return None

        missing = tuple(s for s in INTENT_SPECS[intent].required if s not in state.filled_slots)
        if not missing:
            return None

        value = self._read_slot(missing[0], text, understanding)
        if value is None:
            return None

        state.filled_slots[missing[0]] = value
        state.slot_attempts = 0
        return self._advance(state)

    def _read_slot(self, slot: str, text: str, understanding: Understanding) -> str | None:
        """Try to read one slot value out of the user's reply.

        The NLU's own label is preferred when it has one. Otherwise the whole
        utterance is tried, but only for slots whose values can be CHECKED --
        guessing at an unvalidated slot would fill it with noise.
        """
        for extracted in understanding.slots:
            if extracted.name == slot and self._usable(slot, extracted.value):
                return extracted.value

        cleaned = text.strip()
        if not cleaned or is_affirmative(cleaned) or is_negative(cleaned):
            return None

        if slot == "account_type":
            return cleaned if self.bank.knows_account(cleaned) else None
        if slot == "amount":
            return cleaned if parse_amount(cleaned) is not None else None
        if slot == "card_last4":
            return cleaned if parse_card_last4(cleaned) is not None else None
        if slot == "bill_type":
            return cleaned if self.bank.knows_bill(cleaned) else None
        if slot == "time_period":
            return cleaned if self.bank.knows_period(cleaned) else None
        if slot == "payee":
            # The bot just asked who to send money to; a short reply is a name.
            return cleaned if len(cleaned.split()) <= 4 else None
        return None

    def _advance(self, state: DialogueState) -> BotAction:
        """Ask for the next missing slot, confirm, or execute."""
        intent = state.active_intent
        assert intent is not None
        spec = INTENT_SPECS[intent]

        missing = tuple(s for s in spec.required if s not in state.filled_slots)
        if missing:
            state.slot_attempts += 1
            if state.slot_attempts > self.max_slot_attempts:
                reset(state)
                return BotAction(type=ActionType.GIVE_UP, text=responses.give_up())
            return BotAction(
                type=ActionType.ASK_SLOT,
                text=responses.ask_for(missing[0]),
                intent=intent,
                filled_slots=dict(state.filled_slots),
                missing_slots=missing,
            )

        if spec.confirm:
            state.pending_confirmation = True
            parsed = parse_amount(state.filled_slots.get("amount", ""))
            cents, currency = parsed if parsed else (None, "USD")
            return BotAction(
                type=ActionType.CONFIRM,
                text=responses.confirm(intent, state.filled_slots, cents, currency),
                intent=intent,
                filled_slots=dict(state.filled_slots),
            )

        return self._execute(state)

    def _resolve_confirmation(
        self, text: str, understanding: Understanding, state: DialogueState
    ) -> BotAction:
        """Interpret the answer to a yes/no question."""
        if is_negative(text):
            reset(state)
            return BotAction(type=ActionType.CANCELLED, text=responses.cancelled())

        if is_affirmative(text):
            state.pending_confirmation = False
            return self._execute(state)

        # Neither yes nor no. With barge-in, that is usually a correction --
        # the user cut in to change something rather than to answer. Treat a
        # confident actionable utterance as a revision instead of nagging.
        revision = self._as_revision(understanding, state)
        if revision is not None:
            return revision

        # Genuinely ambiguous: ask again, but do not loop forever.
        state.slot_attempts += 1
        if state.slot_attempts > self.max_slot_attempts:
            reset(state)
            return BotAction(type=ActionType.GIVE_UP, text=responses.give_up())
        return BotAction(
            type=ActionType.CONFIRM,
            text="Sorry — is that a yes or a no?",
            intent=state.active_intent,
            filled_slots=dict(state.filled_slots),
        )

    def _as_revision(self, understanding: Understanding, state: DialogueState) -> BotAction | None:
        """Read an interruption as a change of request, or return None.

        A revision must carry real information: either a different actionable
        intent, or new slot values for the current one. "um" and "hold on" are
        not revisions, and must still be met with the yes/no question.
        """
        if understanding.is_out_of_scope:
            return None
        if understanding.intent_confidence < self.clarify_threshold:
            return None
        if understanding.intent not in INTENT_SPECS:
            return None

        if understanding.intent != state.active_intent:
            reset(state)
            state.active_intent = understanding.intent
            self._merge_slots(understanding, state)
            return self._advance(state)

        if not understanding.slots:
            return None

        # Same intent with new values: re-open the task and reconfirm.
        state.pending_confirmation = False
        self._merge_slots(understanding, state)
        state.slot_attempts = 0
        return self._advance(state)

    def _execute(self, state: DialogueState) -> BotAction:
        """Carry out a fully specified request against the bank."""
        intent = state.active_intent
        assert intent is not None
        slots = dict(state.filled_slots)

        try:
            text = self._run(intent, slots)
        except BankError as exc:
            reset(state)
            return BotAction(
                type=ActionType.FAILED,
                text=responses.failed(str(exc)),
                intent=intent,
                filled_slots=slots,
            )

        reset(state)
        return BotAction(type=ActionType.ANSWER, text=text, intent=intent, filled_slots=slots)

    def _run(self, intent: str, slots: dict[str, str]) -> str:
        """Perform the action. Raises BankError when the bank refuses."""
        if intent == "balance":
            account = self.bank.normalise_account(slots["account_type"])
            return responses.balance_answer(account, self.bank.balance(account))

        if intent == "transfer":
            parsed = parse_amount(slots["amount"])
            if parsed is None:
                raise BankError(f"I couldn't read {slots['amount']!r} as an amount.")
            cents, currency = parsed
            account = self.bank.normalise_account(slots.get("account_type", "checking"))
            remaining = self.bank.transfer(cents, slots["payee"], account)
            return responses.transfer_answer(cents, currency, slots["payee"], account, remaining)

        if intent == "pay_bill":
            bill = slots["bill_type"]
            return responses.bill_answer(bill, self.bank.pay_bill(bill))

        if intent == "spending_history":
            period = slots["time_period"]
            merchant = slots.get("merchant")
            return responses.spending_answer(period, self.bank.spending(period, merchant), merchant)

        if intent in ("freeze_account", "report_fraud"):
            last4 = parse_card_last4(slots["card_last4"])
            if last4 is None:
                raise BankError("I need the last four digits of the card.")
            self.bank.freeze_card(last4)
            if intent == "report_fraud":
                return responses.fraud_answer(last4)
            return responses.freeze_answer(last4)

        if intent == "new_card":
            account = self.bank.normalise_account(slots["account_type"])
            return responses.new_card_answer(account)

        raise BankError(f"I don't know how to carry out {intent!r}.")
