import pytest

from voicebot.nlu.dataset import IGNORE_INDEX, LabelMaps
from voicebot.nlu.slot_templates import slot_tag_vocab
from voicebot.training.evaluate import (
    apply_oos,
    extract_entities,
    intent_metrics,
    slot_metrics,
    tune_oos_threshold,
)


@pytest.fixture
def maps() -> LabelMaps:
    return LabelMaps(intents=["balance", "transfer", "oos"], slot_tags=slot_tag_vocab())


# --- entity extraction -------------------------------------------------------


def test_extract_single_token_entity():
    assert extract_entities(["O", "B-payee", "O"]) == {("payee", 1, 2)}


def test_extract_multi_token_entity():
    assert extract_entities(["B-amount", "I-amount", "I-amount"]) == {("amount", 0, 3)}


def test_two_adjacent_entities_of_the_same_type_stay_separate():
    assert extract_entities(["B-payee", "B-payee"]) == {("payee", 0, 1), ("payee", 1, 2)}


def test_entity_running_to_the_end_of_the_sequence_is_closed():
    assert extract_entities(["O", "B-date", "I-date"]) == {("date", 1, 3)}


def test_stray_inside_tag_opens_a_new_entity():
    # A model can emit I- with no preceding B-; conlleval treats it as a start.
    assert extract_entities(["O", "I-merchant", "O"]) == {("merchant", 1, 2)}


def test_inside_tag_of_a_different_type_starts_a_new_entity():
    assert extract_entities(["B-amount", "I-payee"]) == {("amount", 0, 1), ("payee", 1, 2)}


def test_all_outside_yields_no_entities():
    assert extract_entities(["O", "O", "O"]) == set()


# --- intent metrics ----------------------------------------------------------


def test_intent_metrics_on_a_perfect_prediction(maps):
    metrics = intent_metrics([0, 1, 2], [0, 1, 2], maps)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_intent_metrics_on_a_half_correct_prediction(maps):
    metrics = intent_metrics([0, 0, 1, 1], [0, 1, 1, 0], maps)
    assert metrics["accuracy"] == 0.5


def test_intent_metrics_separate_in_scope_from_out_of_scope(maps):
    # ids: 0 balance, 1 transfer, 2 oos
    metrics = intent_metrics([0, 1, 2, 2], [0, 1, 2, 0], maps)
    assert metrics["in_scope_accuracy"] == 1.0
    assert metrics["oos_class_recall"] == 0.5
    assert metrics["num_in_scope"] == 2
    assert metrics["num_oos"] == 2


# --- slot metrics ------------------------------------------------------------


def test_slot_metrics_ignore_masked_positions(maps):
    gold = [[IGNORE_INDEX, maps.tag_to_id["B-amount"], maps.tag_to_id["I-amount"], IGNORE_INDEX]]
    predicted = [[0, maps.tag_to_id["B-amount"], maps.tag_to_id["I-amount"], 0]]
    metrics = slot_metrics(gold, predicted, maps)
    assert metrics["slot_f1"] == 1.0
    assert metrics["gold_entities"] == 1


def test_a_wrong_boundary_counts_as_a_miss_not_a_partial_credit(maps):
    # gold is a two-token amount; prediction stops one token short
    gold = [[maps.tag_to_id["B-amount"], maps.tag_to_id["I-amount"]]]
    predicted = [[maps.tag_to_id["B-amount"], maps.tag_to_id["O"]]]
    metrics = slot_metrics(gold, predicted, maps)
    assert metrics["slot_f1"] == 0.0


def test_predicting_nothing_gives_zero_recall_and_zero_f1(maps):
    gold = [[maps.tag_to_id["B-payee"]]]
    predicted = [[maps.tag_to_id["O"]]]
    metrics = slot_metrics(gold, predicted, maps)
    assert metrics["slot_recall"] == 0.0
    assert metrics["slot_f1"] == 0.0


def test_slot_metrics_return_zero_when_nothing_is_supervised(maps):
    metrics = slot_metrics([[IGNORE_INDEX, IGNORE_INDEX]], [[0, 0]], maps)
    assert metrics["slot_f1"] == 0.0
    assert metrics["supervised_tokens"] == 0


# --- out-of-scope gate -------------------------------------------------------


def test_tune_threshold_picks_a_separating_value():
    # in-scope predicted correctly and confidently; oos predicted wrongly but weakly
    confidences = [0.95, 0.91, 0.30, 0.22]
    predicted = [0, 1, 0, 1]
    gold = [0, 1, 2, 2]
    threshold, metrics = tune_oos_threshold(confidences, predicted, gold, oos_id=2)
    assert 0.30 < threshold <= 0.91
    assert metrics["oos_handled"] == 1.0
    assert metrics["in_scope_handled"] == 1.0


def test_a_confident_correct_oos_prediction_counts_as_handled():
    """The bug this function was rewritten for.

    A model that confidently and correctly says `oos` has handled the turn.
    Scoring it only by whether the confidence gate fired made more
    out-of-scope training data look like a regression.
    """
    confidences = [0.99, 0.99]  # both far above any sane threshold
    predicted = [2, 2]  # both correctly predicted as oos
    gold = [2, 2]
    _, metrics = tune_oos_threshold(confidences, predicted, gold, oos_id=2)
    assert metrics["oos_handled"] == 1.0


def test_a_correct_but_rejected_in_scope_answer_is_not_handled():
    """A right answer the gate throws away is still a failed turn."""
    confidences = [0.10]
    predicted = [0]
    gold = [0]
    _, metrics = tune_oos_threshold(confidences, predicted, gold, oos_id=2, grid=[0.5])
    assert metrics["in_scope_handled"] == 0.0


def test_tune_threshold_handles_a_set_with_no_oos_examples():
    threshold, metrics = tune_oos_threshold([0.9, 0.8], [0, 1], [0, 1], oos_id=2)
    assert metrics["oos_handled"] == 0.0
    assert 0.0 <= threshold <= 1.0


def test_tune_threshold_without_an_oos_class_scores_in_scope_only():
    threshold, metrics = tune_oos_threshold([0.9, 0.8], [0, 1], [0, 1], oos_id=None)
    assert metrics["in_scope_handled"] == 1.0
    assert 0.0 <= threshold <= 1.0


def test_apply_oos_flags_low_confidence():
    assert apply_oos(0.10, threshold=0.5) is True
    assert apply_oos(0.90, threshold=0.5) is False


def test_zero_confidence_from_empty_input_is_out_of_scope():
    """Empty ASR output yields zero confidence; it must not slip through."""
    assert apply_oos(0.0, threshold=0.5) is True
