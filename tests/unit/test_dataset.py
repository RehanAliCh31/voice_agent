import pytest
import torch

from voicebot.nlu.clinc import TextExample
from voicebot.nlu.dataset import (
    IGNORE_INDEX,
    JointDataset,
    LabelMaps,
    encode_slot_example,
    encode_text_example,
)
from voicebot.nlu.slot_templates import SlotExample, slot_tag_vocab


class StubEncoding:
    """Mimics the parts of transformers' BatchEncoding this module uses."""

    def __init__(self, input_ids, attention_mask, word_ids):
        self.data = {"input_ids": input_ids, "attention_mask": attention_mask}
        self._word_ids = word_ids

    def __getitem__(self, key):
        return self.data[key]

    def word_ids(self, batch_index: int = 0):
        return self._word_ids


class StubTokenizer:
    """One token per word plus [CLS]/[SEP], truncating at max_length."""

    def __call__(
        self,
        words,
        is_split_into_words=False,
        truncation=True,
        max_length=64,
        padding="max_length",
        **kwargs,
    ):
        body = list(words)[: max_length - 2]
        ids = [101] + [1000 + i for i in range(len(body))] + [102]
        word_ids = [None, *range(len(body)), None]
        mask = [1] * len(ids)
        while len(ids) < max_length:
            ids.append(0)
            mask.append(0)
            word_ids.append(None)
        return StubEncoding(ids, mask, word_ids)


@pytest.fixture
def maps() -> LabelMaps:
    return LabelMaps(intents=["balance", "transfer", "oos"], slot_tags=slot_tag_vocab())


def test_text_example_gets_all_ignore_slot_labels(maps):
    item = encode_text_example(
        TextExample(text="what is my balance", intent="balance"),
        StubTokenizer(),
        maps,
        max_length=16,
    )
    assert item["intent_label"].item() == maps.intent_to_id["balance"]
    assert set(item["slot_labels"].tolist()) == {IGNORE_INDEX}


def test_slot_example_labels_only_the_first_subword_of_each_word(maps):
    example = SlotExample(
        text="transfer fifty dollars to sarah",
        intent="transfer",
        tokens=["transfer", "fifty", "dollars", "to", "sarah"],
        tags=["O", "B-amount", "I-amount", "O", "B-payee"],
    )
    item = encode_slot_example(example, StubTokenizer(), maps, max_length=16)
    labels = item["slot_labels"].tolist()
    assert labels[0] == IGNORE_INDEX  # [CLS]
    assert labels[1] == maps.tag_to_id["O"]
    assert labels[2] == maps.tag_to_id["B-amount"]
    assert labels[3] == maps.tag_to_id["I-amount"]
    assert labels[5] == maps.tag_to_id["B-payee"]
    assert labels[6] == IGNORE_INDEX  # [SEP]


def test_truncation_drops_trailing_slot_labels_without_misaligning(maps):
    tokens = ["pay"] * 20 + ["sarah"]
    tags = ["O"] * 20 + ["B-payee"]
    example = SlotExample(text=" ".join(tokens), intent="transfer", tokens=tokens, tags=tags)
    item = encode_slot_example(example, StubTokenizer(), maps, max_length=8)

    labels = item["slot_labels"].tolist()
    assert len(labels) == 8
    assert maps.tag_to_id["B-payee"] not in labels  # the cut-off slot is simply absent
    assert all(label in (IGNORE_INDEX, maps.tag_to_id["O"]) for label in labels)


def test_unknown_intent_raises_a_named_error(maps):
    with pytest.raises(KeyError, match="not_an_intent"):
        encode_text_example(
            TextExample(text="hello", intent="not_an_intent"),
            StubTokenizer(),
            maps,
            max_length=16,
        )


def test_joint_dataset_mixes_both_sources_and_reports_length(maps):
    texts = [TextExample(text="what is my balance", intent="balance")] * 3
    slots = [
        SlotExample(
            text="transfer fifty dollars to sarah",
            intent="transfer",
            tokens=["transfer", "fifty", "dollars", "to", "sarah"],
            tags=["O", "B-amount", "I-amount", "O", "B-payee"],
        )
    ] * 2
    ds = JointDataset(texts, slots, StubTokenizer(), maps, max_length=16)

    assert len(ds) == 5
    item = ds[0]
    assert isinstance(item["input_ids"], torch.Tensor)
    assert item["input_ids"].shape == item["slot_labels"].shape


def test_label_maps_round_trip(maps):
    assert maps.intent_to_id["oos"] == 2
    assert maps.id_to_intent[2] == "oos"
    assert maps.tag_to_id["O"] == 0
