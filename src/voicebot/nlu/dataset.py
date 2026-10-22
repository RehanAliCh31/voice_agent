"""Tokenization, BIO alignment and mixed-supervision batching.

Two corpora feed one model: CLINC150 has intent labels only, the synthetic
corpus has intent plus slot labels. Rather than a separate mask tensor, the
intent-only examples carry slot labels of IGNORE_INDEX (-100), which
torch.nn.functional.cross_entropy skips natively.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.data import Dataset

from voicebot.nlu.clinc import TextExample
from voicebot.nlu.slot_templates import SlotExample

IGNORE_INDEX = -100


@dataclass
class LabelMaps:
    """Label vocabularies and their integer encodings."""

    intents: list[str]
    slot_tags: list[str]
    intent_to_id: dict[str, int] = field(init=False)
    id_to_intent: dict[int, str] = field(init=False)
    tag_to_id: dict[str, int] = field(init=False)
    id_to_tag: dict[int, str] = field(init=False)

    def __post_init__(self) -> None:
        self.intent_to_id = {label: i for i, label in enumerate(self.intents)}
        self.id_to_intent = {i: label for label, i in self.intent_to_id.items()}
        self.tag_to_id = {tag: i for i, tag in enumerate(self.slot_tags)}
        self.id_to_tag = {i: tag for tag, i in self.tag_to_id.items()}

    @property
    def num_intents(self) -> int:
        return len(self.intents)

    @property
    def num_slot_tags(self) -> int:
        return len(self.slot_tags)


def _intent_id(intent: str, maps: LabelMaps) -> int:
    try:
        return maps.intent_to_id[intent]
    except KeyError as exc:
        raise KeyError(
            f"Intent {intent!r} is not in the label vocabulary of {maps.num_intents} intents"
        ) from exc


def encode_text_example(
    example: TextExample, tokenizer: Any, maps: LabelMaps, max_length: int = 64
) -> dict[str, torch.Tensor]:
    """Encode an intent-only example; every slot label is IGNORE_INDEX."""
    intent_id = _intent_id(example.intent, maps)
    encoding = tokenizer(
        example.text.split(),
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    input_ids = encoding["input_ids"]
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
        "intent_label": torch.tensor(intent_id, dtype=torch.long),
        "slot_labels": torch.full((len(input_ids),), IGNORE_INDEX, dtype=torch.long),
    }


def encode_slot_example(
    example: SlotExample, tokenizer: Any, maps: LabelMaps, max_length: int = 64
) -> dict[str, torch.Tensor]:
    """Encode an example with slot labels, aligning BIO tags to subwords.

    Only the first subword of each word carries the tag; continuation subwords
    and special tokens get IGNORE_INDEX. Words cut off by truncation simply
    never appear, so alignment stays consistent.
    """
    intent_id = _intent_id(example.intent, maps)
    encoding = tokenizer(
        example.tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    word_ids = encoding.word_ids(batch_index=0)

    slot_labels: list[int] = []
    previous_word: int | None = None
    for word_id in word_ids:
        if word_id is None:
            slot_labels.append(IGNORE_INDEX)
        elif word_id != previous_word:
            slot_labels.append(maps.tag_to_id[example.tags[word_id]])
        else:
            slot_labels.append(IGNORE_INDEX)
        previous_word = word_id

    return {
        "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
        "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
        "intent_label": torch.tensor(intent_id, dtype=torch.long),
        "slot_labels": torch.tensor(slot_labels, dtype=torch.long),
    }


class JointDataset(Dataset):
    """Interleaves intent-only and slot-labelled examples in one dataset."""

    def __init__(
        self,
        text_examples: Sequence[TextExample],
        slot_examples: Sequence[SlotExample],
        tokenizer: Any,
        maps: LabelMaps,
        max_length: int = 64,
    ) -> None:
        self.text_examples = list(text_examples)
        self.slot_examples = list(slot_examples)
        self.tokenizer = tokenizer
        self.maps = maps
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.text_examples) + len(self.slot_examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < len(self.text_examples):
            return encode_text_example(
                self.text_examples[index], self.tokenizer, self.maps, self.max_length
            )
        offset = index - len(self.text_examples)
        return encode_slot_example(
            self.slot_examples[offset], self.tokenizer, self.maps, self.max_length
        )
