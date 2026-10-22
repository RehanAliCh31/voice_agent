"""The trained joint model, wrapped as an NLUEngine.

This is the bridge between training and the live pipeline. Everything the
training loop learned reaches the bot through here, and nothing else in the
pipeline knows a transformer is involved -- it sees an `NLUEngine` returning
an `Understanding`, exactly like the fake does.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from voicebot.nlu.base import NLUEngine
from voicebot.nlu.bio import ordered_entities, word_spans
from voicebot.nlu.clinc import OOS_LABEL
from voicebot.nlu.dataset import LabelMaps
from voicebot.nlu.model import JointIntentSlotModel
from voicebot.schemas import Slot, Understanding
from voicebot.utils.logging import get_logger
from voicebot.utils.registry import register

log = get_logger("nlu.joint_engine")


@register("nlu", "joint")
class JointNLUEngine(NLUEngine):
    """Runs the fine-tuned encoder and decodes its two heads."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        maps: LabelMaps,
        max_length: int = 64,
        oos_threshold: float = 0.5,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.maps = maps
        self.max_length = max_length
        self.oos_threshold = oos_threshold
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    # --- construction --------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        device: str = "auto",
        oos_threshold: float | None = None,
    ) -> JointNLUEngine:
        """Load a directory written by `voicebot.training.train_intent`.

        The encoder name and the tuned threshold come from the run's own
        metrics.json, so serving cannot silently diverge from training.
        """
        from transformers import AutoTokenizer

        checkpoint_dir = Path(checkpoint_dir)
        labels = json.loads((checkpoint_dir / "label_maps.json").read_text(encoding="utf-8"))
        maps = LabelMaps(intents=labels["intents"], slot_tags=labels["slot_tags"])

        metrics = json.loads((checkpoint_dir / "metrics.json").read_text(encoding="utf-8"))
        config = metrics["config"]
        encoder_name = config["nlu"]["encoder_name"]
        max_length = config["nlu"]["max_length"]
        threshold = (
            oos_threshold if oos_threshold is not None else metrics.get("oos_threshold", 0.5)
        )

        model = JointIntentSlotModel.from_pretrained_encoder(
            encoder_name=encoder_name,
            num_intents=maps.num_intents,
            num_slot_tags=maps.num_slot_tags,
        )
        model.load_state_dict(torch.load(checkpoint_dir / "model.pt", map_location="cpu"))

        tokenizer = AutoTokenizer.from_pretrained(encoder_name, add_prefix_space=True)
        resolved = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        log.info(
            "loaded %s from %s (threshold %.2f) on %s",
            encoder_name,
            checkpoint_dir,
            threshold,
            resolved,
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            maps=maps,
            max_length=max_length,
            oos_threshold=threshold,
            device=resolved,
        )

    # --- inference -----------------------------------------------------------

    @torch.no_grad()
    def understand(self, text: str) -> Understanding:
        """Classify one utterance and extract its slots."""
        # Silence reaches here as an empty transcript. A classifier handed an
        # empty string will still pick something, confidently. Refuse instead.
        if not text.strip():
            return Understanding(
                intent=OOS_LABEL,
                intent_confidence=0.0,
                oos_score=1.0,
                is_out_of_scope=True,
            )

        words = text.split()
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        input_ids = torch.tensor([encoding["input_ids"]], dtype=torch.long, device=self.device)
        attention = torch.tensor([encoding["attention_mask"]], dtype=torch.long, device=self.device)

        out = self.model(input_ids=input_ids, attention_mask=attention)

        intent_probs = torch.softmax(out.intent_logits[0], dim=-1)
        confidence, index = intent_probs.max(dim=-1)
        intent = self.maps.id_to_intent[int(index)]
        confidence_value = float(confidence)

        slots = self._decode_slots(text, words, out.slot_logits[0], encoding)

        # Out of scope either because the model said so, or because it is not
        # confident enough to act. Both are better than a wrong banking action.
        is_oos = intent == OOS_LABEL or confidence_value < self.oos_threshold

        return Understanding(
            intent=intent,
            intent_confidence=confidence_value,
            slots=tuple(slots),
            oos_score=1.0 - confidence_value,
            is_out_of_scope=is_oos,
        )

    def _decode_slots(
        self, text: str, words: list[str], slot_logits: torch.Tensor, encoding: Any
    ) -> list[Slot]:
        """Turn per-token logits into slots carrying character spans."""
        probs = torch.softmax(slot_logits, dim=-1)
        best_prob, best_index = probs.max(dim=-1)
        word_ids = encoding.word_ids(batch_index=0)

        # One tag per WORD, taken from its first subword -- the same position
        # the training labels were placed on.
        tags: list[str] = ["O"] * len(words)
        scores: list[float] = [0.0] * len(words)
        previous: int | None = None
        for position, word_id in enumerate(word_ids):
            if word_id is not None and word_id != previous and word_id < len(words):
                tags[word_id] = self.maps.id_to_tag.get(int(best_index[position]), "O")
                scores[word_id] = float(best_prob[position])
            previous = word_id

        spans = word_spans(text)
        slots: list[Slot] = []
        for name, start_word, end_word in ordered_entities(tags):
            if end_word > len(spans):
                continue
            start_char = spans[start_word][0]
            end_char = spans[end_word - 1][1]
            confidence = sum(scores[start_word:end_word]) / (end_word - start_word)
            slots.append(
                Slot(
                    name=name,
                    value=text[start_char:end_char],
                    start=start_char,
                    end=end_char,
                    confidence=confidence,
                )
            )
        return slots
