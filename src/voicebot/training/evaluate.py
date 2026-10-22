"""Metrics for the joint model, plus out-of-scope threshold tuning.

Slot quality is measured at ENTITY level, not token level. Token accuracy
flatters a slot tagger badly: most tokens are `O`, so a model that predicts
`O` everywhere already scores well. An entity counts as correct only when its
type and both boundaries match exactly.

Implemented here rather than taken from seqeval, which ships source-only with
a build script that fails on Python 3.13.
"""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.metrics import accuracy_score, f1_score  # type: ignore[import-untyped]

from voicebot.nlu.bio import extract_entities
from voicebot.nlu.clinc import OOS_LABEL
from voicebot.nlu.dataset import IGNORE_INDEX, LabelMaps


def intent_metrics(
    gold_ids: Sequence[int], predicted_ids: Sequence[int], maps: LabelMaps
) -> dict[str, float]:
    """Accuracy and macro-F1 over intent ids.

    Also breaks accuracy down by scope. Overall accuracy mixes two very
    different populations: in-scope intents with 100 training examples each,
    and `oos`, which is every other sentence in the language. Reporting only
    the blend hides which one is failing.
    """
    metrics = {
        "accuracy": float(accuracy_score(gold_ids, predicted_ids)),
        "macro_f1": float(f1_score(gold_ids, predicted_ids, average="macro", zero_division=0)),
        "num_examples": float(len(gold_ids)),
    }

    oos_id = maps.intent_to_id.get(OOS_LABEL)
    if oos_id is None:
        return metrics

    in_scope = [(g, p) for g, p in zip(gold_ids, predicted_ids) if g != oos_id]
    out_scope = [(g, p) for g, p in zip(gold_ids, predicted_ids) if g == oos_id]

    if in_scope:
        metrics["in_scope_accuracy"] = sum(g == p for g, p in in_scope) / len(in_scope)
        metrics["num_in_scope"] = float(len(in_scope))
    if out_scope:
        metrics["oos_class_recall"] = sum(g == p for g, p in out_scope) / len(out_scope)
        metrics["num_oos"] = float(len(out_scope))
    return metrics


def slot_metrics(
    gold_label_ids: Sequence[Sequence[int]],
    predicted_label_ids: Sequence[Sequence[int]],
    maps: LabelMaps,
) -> dict[str, float]:
    """Micro-averaged entity-level precision, recall and F1.

    Positions labelled IGNORE_INDEX are dropped before tagging, so padding,
    special tokens and continuation subwords never affect the score.
    """
    true_positives = 0
    num_predicted = 0
    num_gold = 0
    supervised = 0

    for gold_row, predicted_row in zip(gold_label_ids, predicted_label_ids):
        gold_tags: list[str] = []
        predicted_tags: list[str] = []
        for gold_id, predicted_id in zip(gold_row, predicted_row):
            if gold_id == IGNORE_INDEX:
                continue
            supervised += 1
            gold_tags.append(maps.id_to_tag[int(gold_id)])
            predicted_tags.append(maps.id_to_tag.get(int(predicted_id), "O"))

        if not gold_tags:
            continue
        gold_entities = extract_entities(gold_tags)
        predicted_entities = extract_entities(predicted_tags)
        true_positives += len(gold_entities & predicted_entities)
        num_gold += len(gold_entities)
        num_predicted += len(predicted_entities)

    precision = true_positives / num_predicted if num_predicted else 0.0
    recall = true_positives / num_gold if num_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "slot_precision": float(precision),
        "slot_recall": float(recall),
        "slot_f1": float(f1),
        "supervised_tokens": float(supervised),
        "gold_entities": float(num_gold),
    }


def apply_oos(confidence: float, threshold: float) -> bool:
    """True when the prediction should be rejected as out of scope."""
    return confidence < threshold


def tune_oos_threshold(
    confidences: Sequence[float],
    predicted_ids: Sequence[int],
    gold_ids: Sequence[int],
    oos_id: int | None,
    grid: Sequence[float] | None = None,
) -> tuple[float, dict[str, float]]:
    """Tune the rejection threshold on what the user actually experiences.

    There are TWO ways an out-of-scope utterance can be handled correctly: the
    model predicts `oos` outright, or it is unconfident enough to be rejected.
    Scoring only the second punishes the model for being right, which is how an
    earlier version of this function concluded that MORE out-of-scope training
    data made the gate worse. It had not; the metric had.

    An in-scope utterance is handled correctly only when the prediction is
    right AND survives the gate -- a correct answer the gate throws away is
    still a failed turn.

    The chosen threshold maximises the mean of the two rates, which rules out
    the degenerate answers: 1.0 rejects everything, 0.0 rejects nothing.
    """
    if grid is None:
        grid = [index / 100 for index in range(1, 100)]

    rows = list(zip(confidences, predicted_ids, gold_ids))
    out_scope = [r for r in rows if oos_id is not None and r[2] == oos_id]
    in_scope = [r for r in rows if oos_id is None or r[2] != oos_id]

    best_threshold = 0.5
    best_score = -1.0
    best_metrics: dict[str, float] = {"oos_handled": 0.0, "in_scope_handled": 0.0}

    for threshold in grid:
        handled_oos = sum(
            1 for conf, pred, _ in out_scope if pred == oos_id or apply_oos(conf, threshold)
        )
        handled_in = sum(
            1 for conf, pred, gold in in_scope if pred == gold and not apply_oos(conf, threshold)
        )
        oos_rate = handled_oos / len(out_scope) if out_scope else 0.0
        in_rate = handled_in / len(in_scope) if in_scope else 0.0

        score = (oos_rate + in_rate) / 2 if out_scope else in_rate
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = {"oos_handled": oos_rate, "in_scope_handled": in_rate}

    best_metrics["threshold"] = best_threshold
    best_metrics["balanced_score"] = best_score
    return best_threshold, best_metrics
