"""Re-score a trained checkpoint's out-of-scope gate without retraining.

Threshold tuning needs only the model's confidences and predictions, so it is
inference work, not training work: a minute on a GPU instead of twenty. Use
this after changing how the gate is scored, or to try a different objective.

    python scripts/retune_gate.py \
        --checkpoint checkpoints/roberta_oosplus \
        --clinc data/raw/data/data_oos_plus.json

Writes predictions.json beside the checkpoint and updates metrics.json in
place, so later re-tunes need no model at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from voicebot.config import load_config
from voicebot.nlu.clinc import OOS_LABEL, load_clinc150
from voicebot.nlu.dataset import JointDataset, LabelMaps
from voicebot.nlu.model import JointIntentSlotModel
from voicebot.nlu.slot_templates import generate_split
from voicebot.training.evaluate import intent_metrics, slot_metrics, tune_oos_threshold
from voicebot.utils.logging import get_logger
from voicebot.utils.seed import set_seed

log = get_logger("scripts.retune_gate")


@torch.no_grad()
def _infer(model, loader, device, maps):
    """Run the model over a loader and collect everything the gate needs."""
    model.eval()
    confidences: list[float] = []
    predicted: list[int] = []
    gold: list[int] = []
    gold_slots: list[list[int]] = []
    predicted_slots: list[list[int]] = []

    for batch in loader:
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        probabilities = torch.softmax(out.intent_logits, dim=-1)
        best_confidence, best_index = probabilities.max(dim=-1)

        confidences.extend(best_confidence.cpu().tolist())
        predicted.extend(best_index.cpu().tolist())
        gold.extend(batch["intent_label"].tolist())
        gold_slots.extend(batch["slot_labels"].tolist())
        predicted_slots.extend(out.slot_logits.argmax(dim=-1).cpu().tolist())

    metrics = intent_metrics(gold, predicted, maps)
    metrics.update(slot_metrics(gold_slots, predicted_slots, maps))
    return metrics, confidences, predicted, gold


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Directory holding model.pt")
    parser.add_argument("--clinc", required=True, help="CLINC150 JSON used for the run")
    parser.add_argument("--config", default=None, help="Defaults to the config inside metrics.json")
    parser.add_argument("--slot-examples", type=int, default=5000)
    args = parser.parse_args(argv)

    checkpoint = Path(args.checkpoint)
    metrics_path = checkpoint / "metrics.json"
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))

    # The run's own resolved config is stored in metrics.json, which is exactly
    # why it is written there: a re-score cannot drift from the training setup.
    if args.config:
        config = load_config(args.config)
    else:
        from voicebot.config import AppConfig

        config = AppConfig(**saved["config"])

    set_seed(config.seed)
    labels = json.loads((checkpoint / "label_maps.json").read_text(encoding="utf-8"))
    maps = LabelMaps(intents=labels["intents"], slot_tags=labels["slot_tags"])

    splits = load_clinc150(args.clinc)
    _, slot_held = generate_split(
        n_train=int(args.slot_examples * 0.8),
        n_eval=int(args.slot_examples * 0.2),
        seed=config.seed,
    )
    half = len(slot_held) // 2
    slot_val, slot_test = slot_held[:half], slot_held[half:]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.nlu.encoder_name, add_prefix_space=True)
    max_length = config.nlu.max_length

    model = JointIntentSlotModel.from_pretrained_encoder(
        encoder_name=config.nlu.encoder_name,
        num_intents=maps.num_intents,
        num_slot_tags=maps.num_slot_tags,
    )
    state = torch.load(checkpoint / "model.pt", map_location="cpu")
    model.load_state_dict(state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    log.info("loaded %s on %s", config.nlu.encoder_name, device)

    batch_size = config.training.batch_size
    val_loader = DataLoader(
        JointDataset(splits["val"], slot_val, tokenizer, maps, max_length), batch_size=batch_size
    )
    test_loader = DataLoader(
        JointDataset(splits["test"], slot_test, tokenizer, maps, max_length), batch_size=batch_size
    )

    val_metrics, val_conf, val_pred, val_gold = _infer(model, val_loader, device, maps)
    test_metrics, test_conf, test_pred, test_gold = _infer(model, test_loader, device, maps)

    oos_id = maps.intent_to_id.get(OOS_LABEL)
    threshold, gate = tune_oos_threshold(val_conf, val_pred, val_gold, oos_id)

    saved["val"] = val_metrics
    saved["test"] = test_metrics
    saved["oos_threshold"] = threshold
    saved["oos"] = gate
    metrics_path.write_text(json.dumps(saved, indent=2), encoding="utf-8")

    (checkpoint / "predictions.json").write_text(
        json.dumps(
            {
                "oos_id": oos_id,
                "val": {"confidence": val_conf, "predicted": val_pred, "gold": val_gold},
                "test": {"confidence": test_conf, "predicted": test_pred, "gold": test_gold},
            }
        ),
        encoding="utf-8",
    )

    print(json.dumps({"oos_threshold": threshold, "oos": gate, "test": test_metrics}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
