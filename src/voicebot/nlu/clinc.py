"""Load CLINC150 from its canonical data_full.json layout.

The file holds six keys: train/val/test for in-scope utterances and
oos_train/oos_val/oos_test for the out-of-scope set. We merge each OOS split
into its matching in-scope split under a single `oos` label, because the model
treats out-of-scope as something to detect, not a separate corpus.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from voicebot.utils.logging import get_logger

log = get_logger("nlu.clinc")

OOS_LABEL = "oos"
_SPLITS = ("train", "val", "test")
_REQUIRED_KEYS = (*_SPLITS, "oos_train", "oos_val", "oos_test")


@dataclass(frozen=True)
class TextExample:
    """One labelled utterance with no slot annotation."""

    text: str
    intent: str


def _read_payload(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"CLINC150 file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object, got {type(payload).__name__}")
    missing = [key for key in _REQUIRED_KEYS if key not in payload]
    if missing:
        raise ValueError(
            f"{path} is missing required splits {missing}. Expected keys: {list(_REQUIRED_KEYS)}"
        )
    return payload


def _rows_to_examples(rows: list, path: Path, split: str) -> list[TextExample]:
    examples: list[TextExample] = []
    dropped = 0
    for index, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise ValueError(
                f"{path} split {split!r} row {index} is not a [text, label] pair: {row!r}"
            )
        text, label = row
        if not isinstance(text, str) or not isinstance(label, str) or not text.strip():
            dropped += 1
            continue
        examples.append(TextExample(text=text.strip(), intent=label.strip()))
    if dropped:
        log.warning("dropped %d blank or malformed rows from %s", dropped, split)
    return examples


def load_clinc150(path: str | Path, include_oos: bool = True) -> dict[str, list[TextExample]]:
    """Return {'train': [...], 'val': [...], 'test': [...]} with OOS folded in."""
    path = Path(path)
    payload = _read_payload(path)

    splits: dict[str, list[TextExample]] = {}
    for split in _SPLITS:
        examples = _rows_to_examples(payload[split], path, split)
        if include_oos:
            oos_rows = _rows_to_examples(payload[f"oos_{split}"], path, f"oos_{split}")
            examples.extend(TextExample(text=ex.text, intent=OOS_LABEL) for ex in oos_rows)
        splits[split] = examples
        log.info("loaded %s: %d examples", split, len(examples))
    return splits


def intent_labels(splits: dict[str, list[TextExample]]) -> list[str]:
    """Sorted label vocabulary with `oos` forced last, so its index is stable."""
    labels = {ex.intent for examples in splits.values() for ex in examples}
    labels.discard(OOS_LABEL)
    ordered = sorted(labels)
    if any(ex.intent == OOS_LABEL for examples in splits.values() for ex in examples):
        ordered.append(OOS_LABEL)
    return ordered
