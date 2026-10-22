"""Joint intent classification and slot filling on one shared encoder.

The encoder is injected rather than constructed, so unit tests run against a
tiny stub and never download DistilBERT. Use `from_pretrained_encoder` for
real training.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from voicebot.nlu.dataset import IGNORE_INDEX


@dataclass
class JointOutput:
    """Model output. `loss` is None during inference."""

    loss: torch.Tensor | None
    intent_logits: torch.Tensor
    slot_logits: torch.Tensor


class JointIntentSlotModel(nn.Module):
    """Shared encoder, two heads.

    The intent head reads the [CLS] position; the slot head reads every token.
    Slot labels of IGNORE_INDEX are skipped by cross_entropy, which is how
    intent-only examples coexist with slot-labelled ones in a single batch.
    """

    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int,
        num_intents: int,
        num_slot_tags: int,
        dropout: float = 0.1,
        slot_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.intent_head = nn.Linear(hidden_size, num_intents)
        self.slot_head = nn.Linear(hidden_size, num_slot_tags)
        self.num_intents = num_intents
        self.num_slot_tags = num_slot_tags
        self.slot_loss_weight = slot_loss_weight

    @classmethod
    def from_pretrained_encoder(
        cls,
        encoder_name: str,
        num_intents: int,
        num_slot_tags: int,
        dropout: float = 0.1,
        slot_loss_weight: float = 1.0,
    ) -> JointIntentSlotModel:
        """Build with a real Hugging Face encoder. Downloads on first call."""
        from transformers import AutoModel

        encoder = AutoModel.from_pretrained(encoder_name)
        return cls(
            encoder=encoder,
            hidden_size=encoder.config.hidden_size,
            num_intents=num_intents,
            num_slot_tags=num_slot_tags,
            dropout=dropout,
            slot_loss_weight=slot_loss_weight,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        intent_label: torch.Tensor | None = None,
        slot_labels: torch.Tensor | None = None,
    ) -> JointOutput:
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence = self.dropout(encoded.last_hidden_state)

        intent_logits = self.intent_head(sequence[:, 0])
        slot_logits = self.slot_head(sequence)

        loss: torch.Tensor | None = None
        if intent_label is not None:
            loss = F.cross_entropy(intent_logits, intent_label)

            if slot_labels is not None and self.slot_loss_weight > 0:
                # cross_entropy over an all-IGNORE_INDEX target returns NaN, which
                # would silently poison the whole run. Only add the slot term when
                # at least one position is actually supervised.
                if bool((slot_labels != IGNORE_INDEX).any()):
                    slot_loss = F.cross_entropy(
                        slot_logits.reshape(-1, self.num_slot_tags),
                        slot_labels.reshape(-1),
                        ignore_index=IGNORE_INDEX,
                    )
                    loss = loss + self.slot_loss_weight * slot_loss

        return JointOutput(loss=loss, intent_logits=intent_logits, slot_logits=slot_logits)
