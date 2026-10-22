import json

import pytest

from download_data import verify_clinc_json


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_verify_returns_counts_for_each_split(tmp_path):
    path = _write(
        tmp_path / "data_full.json",
        {
            "train": [["what is my balance", "balance"]],
            "val": [["check balance", "balance"]],
            "test": [["balance please", "balance"]],
            "oos_train": [["what is the weather", "oos"]],
            "oos_val": [["tell me a joke", "oos"]],
            "oos_test": [["sing a song", "oos"]],
        },
    )
    counts = verify_clinc_json(path)
    assert counts["train"] == 1
    assert counts["oos_test"] == 1


def test_verify_names_the_file_and_the_missing_keys(tmp_path):
    path = _write(tmp_path / "broken.json", {"train": []})
    with pytest.raises(ValueError) as exc:
        verify_clinc_json(path)
    message = str(exc.value)
    assert "broken.json" in message
    assert "val" in message and "test" in message


def test_verify_rejects_rows_that_are_not_text_label_pairs(tmp_path):
    path = _write(
        tmp_path / "shape.json",
        {
            "train": [["only one element"]],
            "val": [],
            "test": [],
            "oos_train": [],
            "oos_val": [],
            "oos_test": [],
        },
    )
    with pytest.raises(ValueError, match="pair"):
        verify_clinc_json(path)


def test_verify_reports_a_missing_file_by_name(tmp_path):
    with pytest.raises(FileNotFoundError, match="absent"):
        verify_clinc_json(tmp_path / "absent.json")
