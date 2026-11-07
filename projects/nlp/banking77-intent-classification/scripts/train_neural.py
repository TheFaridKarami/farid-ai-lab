"""Train and compare the neural text classifiers on Banking77.

Run from the repository root:
    PYTHONPATH=src python scripts/train_neural.py
    PYTHONPATH=src python scripts/train_neural.py --models gru lstm
"""

from __future__ import annotations

# Import Triton before anything that pulls in TensorFlow (the tokenizer does).
# In the other order the process aborts the first time torch.optim imports
# Triton. Triton is optional, so a missing install is fine.
try:
    import triton  # noqa: F401
except ImportError:
    pass

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from training.neural_trainer import PADDING_IDX, NeuralTrainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "model.yaml"
MODELS_DIR = PROJECT_ROOT / "artifacts" / "neural" / "models"
REPORTS_DIR = PROJECT_ROOT / "reports" / "neural"

MODEL_NAMES = ("embedding_bag_mlp", "rnn", "lstm", "gru")

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------- utilities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and compare the neural text classifiers."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=list(MODEL_NAMES),
        help="Models to train (default: all four).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML config.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def to_json_compatible(value: Any) -> Any:
    """Convert NumPy, PyTorch, and Path values to plain JSON types.

    NaN and infinity become None, so a diverged run still produces a valid
    report instead of crashing after training has finished.
    """
    if isinstance(value, dict):
        return {str(key): to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_compatible(item) for item in value]
    if isinstance(value, torch.Tensor):
        return to_json_compatible(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return to_json_compatible(value.tolist())
    if isinstance(value, np.generic):
        return to_json_compatible(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(data: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            to_json_compatible(data),
            file,
            indent=4,
            ensure_ascii=False,
            allow_nan=False,
        )


def save_checkpoint(trainer: NeuralTrainer, model_name: str) -> Path:
    """Save everything needed to rebuild the model and decode its predictions.

    The checkpoint holds only tensors, lists, strings, and numbers, so it also
    loads under torch.load's default weights_only=True.
    """
    vocabulary = list(trainer.tokenizer.get_vocabulary())
    label_classes = trainer.label_encoder.classes_.tolist()

    checkpoint = {
        "model_name": model_name,
        # CPU copies, so the file loads on machines without a GPU.
        "model_state_dict": {
            name: tensor.detach().cpu()
            for name, tensor in trainer.model.state_dict().items()
        },
        "model_config": trainer.config,
        "vocab_size": len(vocabulary),
        "vocabulary": vocabulary,
        "num_classes": len(label_classes),
        "label_classes": label_classes,
        "padding_idx": PADDING_IDX,
    }

    path = MODELS_DIR / f"{model_name}.pt"
    torch.save(checkpoint, path)
    return path


# --------------------------------------------------------------- experiment


def run_experiment(trainer: NeuralTrainer, model_name: str) -> dict[str, Any]:
    """Train one architecture, score its best weights, and save the outputs."""
    LOGGER.info("Starting experiment: %s", model_name)

    start = time.perf_counter()
    history = trainer.fit(model_name)
    training_time = time.perf_counter() - start

    # fit() restores the best-epoch weights. Evaluate once more so the reported
    # metrics describe exactly the weights that get saved.
    final_metrics = trainer.evaluate()

    best_epoch = min(history, key=lambda record: record["validation"]["loss"])["epoch"]
    parameters = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)

    checkpoint_path = save_checkpoint(trainer, model_name)

    save_json(
        {
            "model": model_name,
            "config": trainer.config,
            "device": str(trainer.device),
            "parameters": parameters,
            "best_epoch": best_epoch,
            "training_time_seconds": training_time,
            "final_metrics": final_metrics,
            "history": history,
        },
        REPORTS_DIR / f"{model_name}_history.json",
    )

    result = {
        "model": model_name,
        "parameters": parameters,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "training_time_seconds": round(training_time, 1),
        "val_loss": final_metrics["loss"],
        "val_accuracy": final_metrics["accuracy"],
        "val_macro_f1": final_metrics["macro_f1"],
        "val_weighted_f1": final_metrics["weighted_f1"],
    }

    LOGGER.info(
        "%s | best epoch %d | val acc %.4f | val macro-F1 %.4f | %.1fs | saved %s",
        model_name,
        best_epoch,
        result["val_accuracy"],
        result["val_macro_f1"],
        training_time,
        checkpoint_path.name,
    )
    return result


def write_comparison(results: list[dict[str, Any]]) -> None:
    """Rank the models by validation macro-F1 and save the table."""
    table = (
        pd.DataFrame(results)
        .sort_values(by="val_macro_f1", ascending=False)
        .reset_index(drop=True)
    )

    table.to_csv(REPORTS_DIR / "neural_model_comparison.csv", index=False)
    save_json(
        {"results": table.to_dict(orient="records")},
        REPORTS_DIR / "neural_model_comparison.json",
    )

    LOGGER.info("Model comparison (validation set):\n%s", table.to_string(index=False))


# --------------------------------------------------------------------- main


def main() -> int:
    args = parse_args()
    setup_logging()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # One trainer means one tokenizer and one split, so every model sees
    # identical data and the comparison is fair.
    trainer = NeuralTrainer(config_path=args.config)
    trainer.prepare_datasets()

    LOGGER.info("Device: %s", trainer.device)
    LOGGER.info("Training samples: %d", len(trainer.train_loader.dataset))
    LOGGER.info("Validation samples: %d", len(trainer.val_loader.dataset))
    LOGGER.info("Number of classes: %d", len(trainer.label_encoder.classes_))

    results: list[dict[str, Any]] = []
    failures: list[str] = []

    for model_name in args.models:
        try:
            results.append(run_experiment(trainer, model_name))
        except Exception:
            # One failing model (for example, out of GPU memory) should not
            # throw away the results of the others.
            LOGGER.exception("Experiment failed: %s", model_name)
            failures.append(model_name)

    if set(args.models) == set(MODEL_NAMES) and results:
        write_comparison(results)
        LOGGER.info("Reports saved to: %s", REPORTS_DIR)
    else:
        # Writing the comparison from a subset would overwrite a full one.
        LOGGER.info("Subset run: comparison files were not written.")

    LOGGER.info("Checkpoints saved to: %s", MODELS_DIR)

    if failures:
        LOGGER.error("Failed experiments: %s", ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())