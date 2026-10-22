import pytest

from voicebot.dialogue.manager import DialogueManager
from voicebot.dialogue.mock_backend import BankError, MockBank, format_money
from voicebot.dialogue.slots import (
    INTENT_SPECS,
    is_affirmative,
    is_negative,
    parse_amount,
    parse_card_last4,
)
from voicebot.dialogue.states import Phase, phase_of
from voicebot.schemas import ActionType, DialogueState, Slot, Understanding


def _u(intent: str, confidence: float = 0.95, oos: bool = False, **slots) -> Understanding:
    """Build an Understanding as the NLU would produce it."""
    return Understanding(
        intent=intent,
        intent_confidence=confidence,
        slots=tuple(Slot(name=name, value=value, start=0, end=0) for name, value in slots.items()),
        is_out_of_scope=oos,
    )


@pytest.fixture
def manager() -> DialogueManager:
    return DialogueManager(bank=MockBank())


@pytest.fixture
def state() -> DialogueState:
    return DialogueState(session_id="test")


# --- amount parsing ----------------------------------------------------------


@pytest.mark.parametrize(
    "text,cents,currency",
    [
        ("fifty dollars", 5000, "USD"),
        ("250 dollars", 25000, "USD"),
        ("a hundred dollars", 10000, "USD"),
        ("thirty five euros", 3500, "EUR"),
        ("twenty pounds", 2000, "GBP"),
        ("five hundred dollars", 50000, "USD"),
        ("1000 rupees", 100000, "PKR"),
    ],
)
def test_parse_amount_handles_spoken_and_written_numbers(text, cents, currency):
    assert parse_amount(text) == (cents, currency)


def test_parse_amount_returns_none_without_a_number():
    assert parse_amount("some money please") is None


def test_parse_card_last4_takes_the_final_four_digits():
    assert parse_card_last4("4417") == "4417"
    assert parse_card_last4("card ending 9021") == "9021"


def test_parse_card_last4_rejects_short_input():
    assert parse_card_last4("12") is None


def test_money_is_formatted_from_integer_cents():
    assert format_money(248_350) == "$2,483.50"


# --- yes / no ----------------------------------------------------------------


def test_affirmative_and_negative_words():
    assert is_affirmative("yes please")
    assert is_negative("no, cancel that")
    assert not is_affirmative("no")


def test_a_sentence_containing_both_is_not_affirmative():
    # "yes, no wait" must not be read as agreement to move money.
    assert not is_affirmative("yes no wait")


# --- the state machine -------------------------------------------------------


def test_a_low_risk_intent_with_its_slot_answers_immediately(manager, state):
    action = manager.handle(
        "what is my checking balance", _u("balance", account_type="checking"), state
    )
    assert action.type is ActionType.ANSWER
    assert "2,483.50" in action.text
    assert phase_of(state) is Phase.IDLE


def test_a_missing_slot_is_asked_for(manager, state):
    action = manager.handle("what is my balance", _u("balance"), state)
    assert action.type is ActionType.ASK_SLOT
    assert action.missing_slots == ("account_type",)
    assert phase_of(state) is Phase.COLLECTING


def test_slots_accumulate_across_turns(manager, state):
    first = manager.handle("transfer money", _u("transfer"), state)
    assert first.type is ActionType.ASK_SLOT
    assert first.missing_slots == ("amount", "payee")

    second = manager.handle("fifty dollars", _u("transfer", amount="fifty dollars"), state)
    assert second.type is ActionType.ASK_SLOT
    assert second.missing_slots == ("payee",)

    third = manager.handle("to sarah", _u("transfer", payee="sarah"), state)
    assert third.type is ActionType.CONFIRM
    assert phase_of(state) is Phase.CONFIRMING


def test_money_moves_only_after_confirmation(manager, state):
    before = manager.bank.balances["checking"]
    manager.handle(
        "transfer fifty dollars to sarah",
        _u("transfer", amount="fifty dollars", payee="sarah"),
        state,
    )
    assert manager.bank.balances["checking"] == before, "money moved before confirmation"

    action = manager.handle("yes", _u("oos", confidence=0.1), state)
    assert action.type is ActionType.ANSWER
    assert manager.bank.balances["checking"] == before - 5000


def test_declining_a_confirmation_changes_nothing(manager, state):
    before = manager.bank.balances["checking"]
    manager.handle(
        "transfer fifty dollars to sarah",
        _u("transfer", amount="fifty dollars", payee="sarah"),
        state,
    )
    action = manager.handle("no", _u("oos", confidence=0.1), state)
    assert action.type is ActionType.CANCELLED
    assert manager.bank.balances["checking"] == before
    assert phase_of(state) is Phase.IDLE


def test_confirmation_is_resolved_before_the_out_of_scope_gate(manager, state):
    """The bare word "yes" classifies as nonsense; it must still be heard."""
    manager.handle(
        "transfer fifty dollars to sarah",
        _u("transfer", amount="fifty dollars", payee="sarah"),
        state,
    )
    # An Understanding that would otherwise be rejected outright
    action = manager.handle("yes", _u("oos", confidence=0.02, oos=True), state)
    assert action.type is ActionType.ANSWER


def test_an_ambiguous_confirmation_is_queried_again(manager, state):
    manager.handle(
        "transfer fifty dollars to sarah",
        _u("transfer", amount="fifty dollars", payee="sarah"),
        state,
    )
    action = manager.handle("maybe later", _u("oos", confidence=0.1), state)
    assert action.type is ActionType.CONFIRM
    assert phase_of(state) is Phase.CONFIRMING


def test_out_of_scope_declines_without_guessing(manager, state):
    action = manager.handle("what is the weather", _u("oos", confidence=0.2, oos=True), state)
    assert action.type is ActionType.OUT_OF_SCOPE
    assert "weather" not in action.text.lower()


def test_low_confidence_asks_for_a_repeat(manager, state):
    action = manager.handle("mumble", _u("transfer", confidence=0.3), state)
    assert action.type is ActionType.CLARIFY


def test_a_recognised_but_unsupported_intent_says_so(manager, state):
    action = manager.handle("what is my credit score", _u("credit_score"), state)
    assert action.type is ActionType.ANSWER
    assert "credit score" in action.text


def test_repeated_failure_to_fill_a_slot_gives_up(manager, state):
    manager.handle("transfer money", _u("transfer"), state)
    for _ in range(manager.max_slot_attempts):
        action = manager.handle("uh", _u("transfer"), state)
    assert action.type is ActionType.GIVE_UP
    assert phase_of(state) is Phase.IDLE


def test_changing_your_mind_switches_the_active_intent(manager, state):
    manager.handle("transfer money", _u("transfer"), state)
    action = manager.handle(
        "actually what is my savings balance", _u("balance", account_type="savings"), state
    )
    assert action.type is ActionType.ANSWER
    assert "19,020.00" in action.text


def test_an_overdraft_is_refused_with_a_reason(manager, state):
    manager.handle(
        "transfer five hundred thousand dollars to sarah",
        _u("transfer", amount="1000000 dollars", payee="sarah"),
        state,
    )
    action = manager.handle("yes", _u("oos", confidence=0.1), state)
    assert action.type is ActionType.FAILED
    assert "isn't enough" in action.text
    assert phase_of(state) is Phase.IDLE


def test_freezing_a_card_marks_it_frozen(manager, state):
    manager.handle("freeze card 4417", _u("freeze_account", card_last4="4417"), state)
    manager.handle("yes", _u("oos", confidence=0.1), state)
    assert manager.bank.cards[0].frozen is True


def test_an_unknown_card_is_refused(manager, state):
    manager.handle("freeze card 0000", _u("freeze_account", card_last4="0000"), state)
    action = manager.handle("yes", _u("oos", confidence=0.1), state)
    assert action.type is ActionType.FAILED
    assert "0000" in action.text


def test_history_records_every_turn(manager, state):
    manager.handle("what is my balance", _u("balance"), state)
    manager.handle("checking", _u("balance", account_type="checking"), state)
    assert len(state.history) == 2
    assert state.history[0].user_text == "what is my balance"


def test_every_actionable_intent_has_a_prompt_for_each_required_slot():
    from voicebot.dialogue.slots import SLOT_PROMPTS

    for intent, spec in INTENT_SPECS.items():
        for slot in spec.required:
            assert slot in SLOT_PROMPTS, f"{intent} needs a prompt for {slot}"


def test_unknown_account_names_are_refused():
    bank = MockBank()
    with pytest.raises(BankError, match="recognise"):
        bank.normalise_account("my crypto wallet")


# --- interrupting a confirmation (barge-in) ----------------------------------


def test_interrupting_a_confirmation_with_a_new_amount_revises_it(manager, state):
    """Barge-in case: the user cuts in to correct, not to answer yes or no."""
    manager.handle(
        "transfer fifty dollars to sarah",
        _u("transfer", amount="fifty dollars", payee="sarah"),
        state,
    )
    action = manager.handle(
        "actually make it twenty dollars",
        _u("transfer", amount="twenty dollars"),
        state,
    )
    assert action.type is ActionType.CONFIRM
    assert "20.00" in action.text
    assert "50.00" not in action.text


def test_a_revised_transfer_moves_the_revised_amount(manager, state):
    before = manager.bank.balances["checking"]
    manager.handle(
        "transfer fifty dollars to sarah",
        _u("transfer", amount="fifty dollars", payee="sarah"),
        state,
    )
    manager.handle("make it twenty dollars", _u("transfer", amount="twenty dollars"), state)
    manager.handle("yes", _u("oos", confidence=0.1), state)
    assert manager.bank.balances["checking"] == before - 2000


def test_interrupting_with_a_different_intent_switches_task(manager, state):
    manager.handle(
        "transfer fifty dollars to sarah",
        _u("transfer", amount="fifty dollars", payee="sarah"),
        state,
    )
    action = manager.handle(
        "no wait what's my savings balance", _u("balance", account_type="savings"), state
    )
    # "no wait" is a refusal first and foremost; money must not move.
    assert action.type is ActionType.CANCELLED


def test_switching_intent_mid_confirmation_without_a_refusal_word(manager, state):
    manager.handle(
        "transfer fifty dollars to sarah",
        _u("transfer", amount="fifty dollars", payee="sarah"),
        state,
    )
    action = manager.handle(
        "what is my savings balance", _u("balance", account_type="savings"), state
    )
    assert action.type is ActionType.ANSWER
    assert "19,020.00" in action.text


def test_a_meaningless_interruption_still_asks_yes_or_no(manager, state):
    manager.handle(
        "transfer fifty dollars to sarah",
        _u("transfer", amount="fifty dollars", payee="sarah"),
        state,
    )
    action = manager.handle("um hold on", _u("oos", confidence=0.1, oos=True), state)
    assert action.type is ActionType.CONFIRM
    assert "yes or a no" in action.text


def test_a_revision_never_executes_without_a_fresh_confirmation(manager, state):
    before = manager.bank.balances["checking"]
    manager.handle(
        "transfer fifty dollars to sarah",
        _u("transfer", amount="fifty dollars", payee="sarah"),
        state,
    )
    manager.handle("make it twenty dollars", _u("transfer", amount="twenty dollars"), state)
    assert manager.bank.balances["checking"] == before, "revision executed without confirming"
