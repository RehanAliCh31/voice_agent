import json
from pathlib import Path

import pytest

from voicebot.nlu.clinc import OOS_LABEL, TextExample, intent_labels, load_clinc150

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "clinc_tiny.json"


def test_load_merges_oos_into_each_split():
    splits = load_clinc150(FIXTURE)
    assert len(splits["train"]) == 3  # 2 in-scope + 1 oos
    assert any(ex.intent == OOS_LABEL for ex in splits["train"])


def test_load_can_exclude_oos():
    splits = load_clinc150(FIXTURE, include_oos=False)
    assert len(splits["train"]) == 2
    assert all(ex.intent != OOS_LABEL for ex in splits["train"])


def test_examples_are_text_label_dataclasses():
    splits = load_clinc150(FIXTURE)
    first = splits["val"][0]
    assert isinstance(first, TextExample)
    assert first.text == "check my balance"
    assert first.intent == "balance"


def test_intent_labels_are_sorted_with_oos_last():
    labels = intent_labels(load_clinc150(FIXTURE))
    assert labels[-1] == OOS_LABEL
    assert labels[:-1] == sorted(labels[:-1])


def test_missing_file_names_the_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="absent"):
        load_clinc150(tmp_path / "absent.json")


def test_malformed_file_names_the_missing_splits(tmp_path):
    bad = tmp_path / "partial.json"
    bad.write_text(json.dumps({"train": [["a", "b"]]}), encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        load_clinc150(bad)
    assert "partial" in str(exc.value)
    assert "val" in str(exc.value)


def test_blank_text_rows_are_dropped_not_crashed(tmp_path):
    noisy = tmp_path / "noisy.json"
    noisy.write_text(
        json.dumps(
            {
                "train": [["   ", "balance"], ["real text", "balance"]],
                "val": [],
                "test": [],
                "oos_train": [],
                "oos_val": [],
                "oos_test": [],
            }
        ),
        encoding="utf-8",
    )
    splits = load_clinc150(noisy)
    assert len(splits["train"]) == 1
    assert splits["train"][0].text == "real text"
