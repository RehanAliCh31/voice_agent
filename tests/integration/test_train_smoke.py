"""Runs the real training loop end to end with a stub encoder and tokenizer.

Proves the loop wires together correctly without downloading DistilBERT, so it
runs offline and on CPU in a couple of seconds.
"""

import json
from pathlib import Path

import torch
from torch import nn

from voicebot.config import AppConfig
from voicebot.training.train_intent import run_training

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "clinc_tiny.json"
HIDDEN = 16


class StubEncoding:
    def __init__(self, input_ids, attention_mask, word_ids):
        self.data = {"input_ids": input_ids, "attention_mask": attention_mask}
        self._word_ids = word_ids

    def __getitem__(self, key):
        return self.data[key]

    def word_ids(self, batch_index: int = 0):
        return self._word_ids


class StubTokenizer:
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
        ids = [101] + [(hash(w) % 90) + 5 for w in body] + [102]
        word_ids = [None, *range(len(body)), None]
        mask = [1] * len(ids)
        while len(ids) < max_length:
            ids.append(0)
            mask.append(0)
            word_ids.append(None)
        return StubEncoding(ids, mask, word_ids)


class StubEncoder(nn.Module):
    # vocab_size must exceed the ids StubTokenizer emits, which include BERT's
    # real [CLS]=101 and [SEP]=102.
    def __init__(self, hidden_size: int = HIDDEN, vocab_size: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids, attention_mask=None):
        class Output:
            pass

        out = Output()
        out.last_hidden_state = self.embedding(input_ids)
        return out


def _config() -> AppConfig:
    config = AppConfig()
    config.training.epochs = 2
    config.training.batch_size = 4
    config.training.early_stopping_patience = 5
    config.training.device = "cpu"
    config.nlu.max_length = 16
    return config


def test_training_runs_and_writes_metrics_and_checkpoint(tmp_path):
    metrics = run_training(
        config=_config(),
        clinc_path=FIXTURE,
        output_dir=tmp_path,
        num_slot_examples=40,
        tokenizer=StubTokenizer(),
        encoder_factory=lambda: (StubEncoder(), HIDDEN),
    )

    assert "val" in metrics and "test" in metrics
    assert 0.0 <= metrics["test"]["accuracy"] <= 1.0
    assert "oos_threshold" in metrics
    assert metrics["seed"] == 42

    written = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert written["test"]["accuracy"] == metrics["test"]["accuracy"]
    assert "config" in written

    assert (tmp_path / "model.pt").is_file()
    labels = json.loads((tmp_path / "label_maps.json").read_text(encoding="utf-8"))
    assert "oos" in labels["intents"]


def test_training_loss_is_finite_throughout(tmp_path):
    metrics = run_training(
        config=_config(),
        clinc_path=FIXTURE,
        output_dir=tmp_path,
        num_slot_examples=40,
        tokenizer=StubTokenizer(),
        encoder_factory=lambda: (StubEncoder(), HIDDEN),
    )
    assert len(metrics["epochs"]) >= 1
    for epoch in metrics["epochs"]:
        assert torch.isfinite(torch.tensor(epoch["train_loss"])), epoch


def test_saved_checkpoint_can_be_loaded_back(tmp_path):
    run_training(
        config=_config(),
        clinc_path=FIXTURE,
        output_dir=tmp_path,
        num_slot_examples=40,
        tokenizer=StubTokenizer(),
        encoder_factory=lambda: (StubEncoder(), HIDDEN),
    )
    state = torch.load(tmp_path / "model.pt", map_location="cpu")
    assert "intent_head.weight" in state
    assert "slot_head.weight" in state
