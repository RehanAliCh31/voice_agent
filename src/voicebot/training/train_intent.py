"""Train the joint intent and slot model.

Entry point:

    python -m voicebot.training.train_intent \
        --config configs/intent_distilbert.yaml \
        --clinc data/raw/data/data_full.json \
        --output checkpoints/intent_v1

`tokenizer` and `encoder_factory` are injectable so the integration test can
exercise this exact loop with stubs, offline and without a GPU.

A note on what the slot score means: the synthetic corpus is split 80/20, so
held-out slot examples reuse the same TEMPLATES with different values. Slot F1
therefore measures generalisation across slot values, not across phrasings.
That limitation belongs in the model card.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from voicebot.config import AppConfig, load_config
from voicebot.nlu.baseline_svm import train_baseline
from voicebot.nlu.clinc import OOS_LABEL, intent_labels, load_clinc150
from voicebot.nlu.dataset import JointDataset, LabelMaps
from voicebot.nlu.model import JointIntentSlotModel
from voicebot.nlu.slot_templates import generate_split, slot_tag_vocab
from voicebot.training.evaluate import intent_metrics, slot_metrics, tune_oos_threshold
from voicebot.utils.logging import get_logger
from voicebot.utils.seed import set_seed

log = get_logger("training.train_intent")


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _default_tokenizer(encoder_name: str) -> Any:
    from transformers import AutoTokenizer

    # add_prefix_space is required by RoBERTa-family BPE tokenizers when the
    # input is already split into words; BERT-family tokenizers ignore it.
    return AutoTokenizer.from_pretrained(encoder_name, add_prefix_space=True)


@torch.no_grad()
def _evaluate(
    model: JointIntentSlotModel,
    loader: DataLoader,
    device: torch.device,
    maps: LabelMaps,
) -> tuple[dict[str, float], list[float], list[int], list[int]]:
    """Return (metrics, confidence, predicted intent ids, gold intent ids)."""
    model.eval()
    gold_intents: list[int] = []
    predicted_intents: list[int] = []
    confidences: list[float] = []
    gold_slots: list[list[int]] = []
    predicted_slots: list[list[int]] = []

    for batch in loader:
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        probabilities = torch.softmax(out.intent_logits, dim=-1)
        best_confidence, best_index = probabilities.max(dim=-1)

        gold_intents.extend(batch["intent_label"].tolist())
        predicted_intents.extend(best_index.cpu().tolist())
        confidences.extend(best_confidence.cpu().tolist())

        gold_slots.extend(batch["slot_labels"].tolist())
        predicted_slots.extend(out.slot_logits.argmax(dim=-1).cpu().tolist())

    metrics = intent_metrics(gold_intents, predicted_intents, maps)
    metrics.update(slot_metrics(gold_slots, predicted_slots, maps))
    return metrics, confidences, predicted_intents, gold_intents


def run_training(
    config: AppConfig,
    clinc_path: str | Path,
    output_dir: str | Path,
    num_slot_examples: int = 5000,
    tokenizer: Any | None = None,
    encoder_factory: Callable[[], tuple[torch.nn.Module, int]] | None = None,
    run_baseline: bool = False,
) -> dict:
    """Train, evaluate, and write model.pt / label_maps.json / metrics.json."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)

    splits = load_clinc150(clinc_path)
    # Split by template and value, never by example: see generate_split.
    slot_train, slot_held = generate_split(
        n_train=int(num_slot_examples * 0.8),
        n_eval=int(num_slot_examples * 0.2),
        seed=config.seed,
    )
    slot_corpus = [*slot_train, *slot_held]

    # The label space must cover BOTH corpora. Today every template intent also
    # exists in CLINC150, but nothing enforces that, so build the union rather
    # than depending on it silently -- and say so when a template intent has no
    # real examples behind it.
    clinc_intents = intent_labels(splits)
    synthetic_intents = {ex.intent for ex in slot_corpus}
    synthetic_only = sorted(synthetic_intents - set(clinc_intents))
    if synthetic_only:
        log.warning(
            "template intents absent from CLINC150: %s - these classes will be "
            "trained on synthetic data only",
            synthetic_only,
        )
    in_scope = sorted((set(clinc_intents) | synthetic_intents) - {OOS_LABEL})
    merged = [*in_scope, OOS_LABEL] if OOS_LABEL in clinc_intents else in_scope

    maps = LabelMaps(intents=merged, slot_tags=slot_tag_vocab())
    log.info("intents=%d slot_tags=%d", maps.num_intents, maps.num_slot_tags)

    half = len(slot_held) // 2
    slot_val, slot_test = slot_held[:half], slot_held[half:]

    tokenizer = tokenizer or _default_tokenizer(config.nlu.encoder_name)
    max_length = config.nlu.max_length

    train_ds = JointDataset(splits["train"], slot_train, tokenizer, maps, max_length)
    val_ds = JointDataset(splits["val"], slot_val, tokenizer, maps, max_length)
    test_ds = JointDataset(splits["test"], slot_test, tokenizer, maps, max_length)

    batch_size = config.training.batch_size
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    if encoder_factory is not None:
        encoder, hidden_size = encoder_factory()
        model = JointIntentSlotModel(
            encoder=encoder,
            hidden_size=hidden_size,
            num_intents=maps.num_intents,
            num_slot_tags=maps.num_slot_tags,
            dropout=config.training.dropout,
            slot_loss_weight=config.training.slot_loss_weight,
        )
    else:
        model = JointIntentSlotModel.from_pretrained_encoder(
            encoder_name=config.nlu.encoder_name,
            num_intents=maps.num_intents,
            num_slot_tags=maps.num_slot_tags,
            dropout=config.training.dropout,
            slot_loss_weight=config.training.slot_loss_weight,
        )

    device = _resolve_device(config.training.device)
    model.to(device)
    log.info("device=%s train_batches=%d", device, len(train_loader))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    total_steps = max(1, len(train_loader) * config.training.epochs)
    warmup_steps = int(total_steps * config.training.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return step / max(1, warmup_steps)
        remaining = total_steps - warmup_steps
        return max(0.0, (total_steps - step) / max(1, remaining))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    epoch_records: list[dict] = []
    best_macro_f1 = -1.0
    best_state: dict | None = None
    patience_left = config.training.early_stopping_patience

    for epoch in range(1, config.training.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                intent_label=batch["intent_label"].to(device),
                slot_labels=batch["slot_labels"].to(device),
            )
            assert out.loss is not None
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += float(out.loss.item())

        train_loss = running_loss / max(1, len(train_loader))
        val_metrics, _, _, _ = _evaluate(model, val_loader, device, maps)
        log.info(
            "epoch %d train_loss=%.4f val_acc=%.4f val_macro_f1=%.4f val_slot_f1=%.4f",
            epoch,
            train_loss,
            val_metrics["accuracy"],
            val_metrics["macro_f1"],
            val_metrics["slot_f1"],
        )
        epoch_records.append({"epoch": epoch, "train_loss": train_loss, "val": val_metrics})

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = config.training.early_stopping_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                log.info("early stopping at epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_metrics, val_conf, val_pred, val_gold = _evaluate(model, val_loader, device, maps)
    oos_id = maps.intent_to_id.get(OOS_LABEL)
    threshold, threshold_metrics = tune_oos_threshold(val_conf, val_pred, val_gold, oos_id)

    test_metrics, test_conf, test_pred, test_gold = _evaluate(model, test_loader, device, maps)

    baseline_metrics = None
    if run_baseline:
        _, baseline_metrics = train_baseline(splits["train"], splits["test"])

    metrics = {
        "seed": config.seed,
        "device": str(device),
        "epochs": epoch_records,
        "val": val_metrics,
        "test": test_metrics,
        "oos_threshold": threshold,
        "oos": threshold_metrics,
        "baseline": baseline_metrics,
        "config": config.model_dump(),
    }

    torch.save(model.state_dict(), output_dir / "model.pt")
    (output_dir / "label_maps.json").write_text(
        json.dumps({"intents": maps.intents, "slot_tags": maps.slot_tags}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    # Raw predictions, so the gate can be re-tuned later without a GPU.
    (output_dir / "predictions.json").write_text(
        json.dumps(
            {
                "oos_id": oos_id,
                "val": {"confidence": val_conf, "predicted": val_pred, "gold": val_gold},
                "test": {"confidence": test_conf, "predicted": test_pred, "gold": test_gold},
            }
        ),
        encoding="utf-8",
    )
    log.info("wrote checkpoint and metrics to %s", output_dir)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the joint intent and slot model.")
    parser.add_argument("--config", default="configs/intent_distilbert.yaml")
    parser.add_argument("--clinc", default="data/raw/data/data_full.json")
    parser.add_argument("--output", default="checkpoints/intent_v1")
    parser.add_argument("--slot-examples", type=int, default=5000)
    parser.add_argument("--baseline", action="store_true", help="Also train the SVM baseline")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    metrics = run_training(
        config=config,
        clinc_path=args.clinc,
        output_dir=args.output,
        num_slot_examples=args.slot_examples,
        run_baseline=args.baseline,
    )
    print(
        json.dumps(
            {"test": metrics["test"], "oos_threshold": metrics["oos_threshold"]},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
