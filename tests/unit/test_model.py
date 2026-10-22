import torch
from torch import nn

from voicebot.nlu.dataset import IGNORE_INDEX
from voicebot.nlu.model import JointIntentSlotModel

HIDDEN = 16
INTENTS = 5
TAGS = 7


class StubEncoder(nn.Module):
    """Stands in for DistilBERT: embeds token ids and returns last_hidden_state."""

    def __init__(self, hidden_size: int = HIDDEN, vocab_size: int = 100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids, attention_mask=None):
        class Output:
            pass

        out = Output()
        out.last_hidden_state = self.embedding(input_ids)
        return out


def _model(slot_loss_weight: float = 1.0) -> JointIntentSlotModel:
    torch.manual_seed(0)
    model = JointIntentSlotModel(
        encoder=StubEncoder(),
        hidden_size=HIDDEN,
        num_intents=INTENTS,
        num_slot_tags=TAGS,
        slot_loss_weight=slot_loss_weight,
    )
    # eval() disables dropout so two forward passes over the same batch are
    # comparable. Backward still works, which the gradient test relies on.
    model.eval()
    return model


def _batch(batch_size: int = 4, length: int = 6):
    return {
        "input_ids": torch.randint(0, 100, (batch_size, length)),
        "attention_mask": torch.ones(batch_size, length, dtype=torch.long),
    }


def test_forward_without_labels_returns_logits_of_the_right_shape():
    out = _model()(**_batch())
    assert out.intent_logits.shape == (4, INTENTS)
    assert out.slot_logits.shape == (4, 6, TAGS)
    assert out.loss is None


def test_loss_is_finite_when_both_label_types_are_present():
    batch = _batch()
    slot_labels = torch.full((4, 6), IGNORE_INDEX)
    slot_labels[0, 1] = 2
    slot_labels[1, 3] = 4
    out = _model()(
        **batch,
        intent_label=torch.randint(0, INTENTS, (4,)),
        slot_labels=slot_labels,
    )
    assert torch.isfinite(out.loss)
    assert out.loss.item() > 0


def test_all_ignore_slot_labels_do_not_produce_nan_loss():
    """A batch of intent-only examples must still train."""
    batch = _batch()
    out = _model()(
        **batch,
        intent_label=torch.randint(0, INTENTS, (4,)),
        slot_labels=torch.full((4, 6), IGNORE_INDEX),
    )
    assert torch.isfinite(out.loss), "slot loss over all -100 labels leaked NaN into the total"


def test_intent_only_batch_loss_equals_the_intent_loss_alone():
    batch = _batch()
    intent_label = torch.randint(0, INTENTS, (4,))
    model = _model()
    with_slots_ignored = model(
        **batch, intent_label=intent_label, slot_labels=torch.full((4, 6), IGNORE_INDEX)
    )
    intent_only = model(**batch, intent_label=intent_label)
    assert torch.allclose(with_slots_ignored.loss, intent_only.loss)


def test_slot_loss_weight_scales_the_total_loss():
    batch = _batch()
    intent_label = torch.randint(0, INTENTS, (4,))
    slot_labels = torch.full((4, 6), IGNORE_INDEX)
    slot_labels[:, 1] = 3

    light = _model(slot_loss_weight=0.0)(
        **batch, intent_label=intent_label, slot_labels=slot_labels
    )
    heavy = _model(slot_loss_weight=2.0)(
        **batch, intent_label=intent_label, slot_labels=slot_labels
    )
    assert heavy.loss.item() > light.loss.item()


def test_gradients_flow_to_both_heads():
    batch = _batch()
    slot_labels = torch.full((4, 6), IGNORE_INDEX)
    slot_labels[:, 2] = 1
    model = _model()
    out = model(**batch, intent_label=torch.randint(0, INTENTS, (4,)), slot_labels=slot_labels)
    out.loss.backward()
    assert model.intent_head.weight.grad is not None
    assert model.slot_head.weight.grad is not None
    assert torch.isfinite(model.slot_head.weight.grad).all()
