# scripts/train_transformer.py
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch._dynamo
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # repo imports use `from models...`, not `from src.models...`

from data.data import prepare_data, split_data
from evaluation.evaluator import ClassificationEvaluator
from training.trainer_transformer import TransformerTrainer

logger = logging.getLogger("train_transformer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the from-scratch Transformer on Banking77.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "model.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "transformer")
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Also score the test set. Use once, for the final model comparison.",
    )
    return parser.parse_args()


def evaluate(name, evaluator, y_true, y_pred) -> dict:
    metrics = {key: float(value) for key, value in evaluator.compute_metrics(y_true, y_pred).items()}
    logger.info(
        "%s | accuracy %.4f | macro F1 %.4f | weighted F1 %.4f",
        name, metrics["accuracy"], metrics["macro_f1"], metrics["weighted_f1"],
    )
    return metrics


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    config = yaml.safe_load(args.config.read_text())

    # --- data: your own pipeline ---
    train_df, test_df = prepare_data()
    X_train, X_val, y_train, y_val = split_data(
        train_df, random_state=config["transformer"]["seed"]
    )
    logger.info("train %d | val %d | test %d", len(X_train), len(X_val), len(test_df))

    # --- tokenizer, label encoder, loaders ---
    trainer = TransformerTrainer(config)
    tokenizer, label_encoder, train_loader, val_loader = trainer.prepare_data(
        X_train, X_val, y_train, y_val
    )
    vocabulary = tokenizer.get_vocabulary()
    class_names = list(label_encoder.classes_)

    # --- train ---
    model, history = trainer.train_model(
        train_loader,
        val_loader,
        vocab_size=len(vocabulary),
        num_classes=len(class_names),
    )

    # --- evaluate ---
    evaluator = ClassificationEvaluator(
        class_ids=list(range(len(class_names))), class_names=class_names
    )
    y_val_ids = label_encoder.transform(y_val)
    y_val_pred = trainer.predict(model, val_loader)

    results = {"validation": evaluate("validation", evaluator, y_val_ids, y_val_pred)}
    report = evaluator.build_report(y_val_ids, y_val_pred)

    for true_name, predicted_name, count in evaluator.most_confused_pairs(y_val_ids, y_val_pred):
        logger.info("confused: %s -> %s (%d)", true_name, predicted_name, count)

    if args.evaluate_test:
        test_loader = trainer.build_loader(
            tokenizer, label_encoder, test_df["text"], test_df["label"]
        )
        y_test_ids = label_encoder.transform(test_df["label"])
        y_test_pred = trainer.predict(model, test_loader)
        results["test"] = evaluate("test", evaluator, y_test_ids, y_test_pred)
        report = evaluator.build_report(y_test_ids, y_test_pred)

    # --- save everything the predictor will need ---
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocabulary": vocabulary,
            "class_names": class_names,
            "config": config["transformer"],
        },
        args.output_dir / "model.pt",
    )
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))
    (args.output_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    (args.output_dir / "classification_report.txt").write_text(report)
    logger.info("saved artifacts to %s", args.output_dir)


if __name__ == "__main__":
    main()