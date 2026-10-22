"""The trained NLU engine, driven by a scripted model so no download happens."""

import pytest
import torch

from voicebot.nlu.bio import word_spans
from voicebot.nlu.dataset import LabelMaps
from voicebot.nlu.joint_engine import JointNLUEngine
from voicebot.nlu.slot_templates import slot_tag_vocab
from voicebot.schemas import Understanding

INTENTS = ["balance", "transfer", "oos"]


class StubEncoding:
    def __init__(self, input_ids, attention_mask, word_ids):
        self.data = {"input_ids": input_ids, "attention_mask": attention_mask}
        self._word_ids = word_ids

    def __getitem__(self, key):
        return self.data[key]

    def word_ids(self, batch_index: int = 0):
        return self._word_ids


class StubTokenizer:
    """One token per word plus [CLS]/[SEP]."""

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


class ScriptedModel(torch.nn.Module):
    """Returns the intent and per-word tags it was told to.

    Driving the engine with a real network would make these tests depend on
    what the network happens to have learned. Here the prediction is the
    fixture, so the tests check the DECODING, which is what this class does.
    """

    def __init__(self, maps: LabelMaps, intent: str, confidence: float, word_tags: list[str]):
        super().__init__()
        self.maps = maps
        self.intent = intent
        self.confidence = confidence
        self.word_tags = word_tags

    def to(self, device):
        return self

    def eval(self):
        return self

    def forward(self, input_ids, attention_mask=None):
        batch, length = input_ids.shape
        # Intent logits shaped so softmax gives roughly the asked-for confidence
        intent_logits = torch.zeros(batch, self.maps.num_intents)
        target = self.maps.intent_to_id[self.intent]
        intent_logits[0, target] = self._logit_for(self.confidence, self.maps.num_intents)

        slot_logits = torch.zeros(batch, length, self.maps.num_slot_tags)
        slot_logits[:, :, self.maps.tag_to_id["O"]] = 5.0
        # position 0 is [CLS]; word i sits at position i + 1
        for word_index, tag in enumerate(self.word_tags):
            position = word_index + 1
            if position < length:
                slot_logits[0, position, self.maps.tag_to_id["O"]] = 0.0
                slot_logits[0, position, self.maps.tag_to_id[tag]] = 8.0

        class Output:
            pass

        out = Output()
        out.intent_logits = intent_logits
        out.slot_logits = slot_logits
        return out

    @staticmethod
    def _logit_for(confidence: float, num_classes: int) -> float:
        """Logit that makes softmax produce approximately `confidence`."""
        import math

        confidence = min(max(confidence, 1e-6), 1 - 1e-6)
        return math.log(confidence * (num_classes - 1) / (1 - confidence))


def _engine(intent: str, confidence: float, word_tags: list[str], threshold: float = 0.5):
    maps = LabelMaps(intents=INTENTS, slot_tags=slot_tag_vocab())
    return JointNLUEngine(
        model=ScriptedModel(maps, intent, confidence, word_tags),
        tokenizer=StubTokenizer(),
        maps=maps,
        max_length=32,
        oos_threshold=threshold,
    )


# --- character spans ---------------------------------------------------------


def test_word_spans_cover_the_original_text():
    text = "transfer fifty dollars to sarah"
    spans = word_spans(text)
    assert [text[a:b] for a, b in spans] == text.split()


def test_word_spans_survive_repeated_words():
    text = "pay pay pay the bill"
    spans = word_spans(text)
    assert [text[a:b] for a, b in spans] == ["pay", "pay", "pay", "the", "bill"]
    assert spans[0] != spans[1]


# --- decoding ----------------------------------------------------------------


def test_intent_and_confidence_are_returned():
    engine = _engine("balance", 0.93, ["O", "O", "O"])
    result = engine.understand("what is balance")
    assert isinstance(result, Understanding)
    assert result.intent == "balance"
    assert result.intent_confidence == pytest.approx(0.93, abs=0.02)


def test_a_multi_word_slot_becomes_one_slot_with_the_full_span():
    text = "transfer fifty dollars to sarah"
    engine = _engine("transfer", 0.95, ["O", "B-amount", "I-amount", "O", "B-payee"])
    result = engine.understand(text)

    names = {s.name: s for s in result.slots}
    assert set(names) == {"amount", "payee"}
    assert names["amount"].value == "fifty dollars"
    assert names["payee"].value == "sarah"
    # spans must index back into the original string
    assert text[names["amount"].start : names["amount"].end] == "fifty dollars"


def test_slots_come_back_in_reading_order():
    engine = _engine("transfer", 0.95, ["O", "B-amount", "I-amount", "O", "B-payee"])
    result = engine.understand("transfer fifty dollars to sarah")
    assert [s.name for s in result.slots] == ["amount", "payee"]


def test_no_slots_when_every_tag_is_outside():
    engine = _engine("balance", 0.9, ["O", "O", "O"])
    assert engine.understand("what is balance").slots == ()


def test_slot_confidence_is_reported():
    engine = _engine("transfer", 0.95, ["O", "B-payee"])
    slot = engine.understand("pay sarah").slots[0]
    assert 0.0 < slot.confidence <= 1.0


# --- the out-of-scope gate ---------------------------------------------------


def test_predicting_oos_marks_the_utterance_out_of_scope():
    engine = _engine("oos", 0.88, ["O", "O"])
    result = engine.understand("the weather")
    assert result.is_out_of_scope is True


def test_low_confidence_marks_the_utterance_out_of_scope():
    engine = _engine("transfer", 0.20, ["O", "O"], threshold=0.5)
    result = engine.understand("mumble mumble")
    assert result.is_out_of_scope is True


def test_a_confident_in_scope_prediction_passes_the_gate():
    engine = _engine("transfer", 0.97, ["O", "O"], threshold=0.5)
    assert engine.understand("send money").is_out_of_scope is False


def test_oos_score_is_the_complement_of_confidence():
    engine = _engine("balance", 0.90, ["O", "O"])
    result = engine.understand("my balance")
    assert result.oos_score == pytest.approx(1.0 - result.intent_confidence)


def test_empty_input_is_out_of_scope_without_running_the_model():
    engine = _engine("transfer", 0.99, ["O"])
    result = engine.understand("   ")
    assert result.is_out_of_scope is True
    assert result.intent_confidence == 0.0
    assert result.slots == ()
