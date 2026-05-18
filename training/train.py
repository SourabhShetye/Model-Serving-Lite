"""
training/train.py

Fine-tuning script for the sentiment classifier.

Design decisions:
  1. Self-contained — reads from training/data/, writes to --output-dir.
     No database, no Redis, no FastAPI imports. The CI workflow calls this
     as a subprocess; it should have zero coupling to the serving layer.

  2. Adapter-style fine-tuning — we freeze the base DistilBERT layers and
     only train the classification head. This makes training feasible on
     CPU in CI (~10-15 min vs ~6 hours for full fine-tuning).

  3. Outputs a metrics.json alongside the model weights so evaluate.py
     can verify the training run completed successfully before the
     promote-gate reads it.

  4. Deterministic — fixed random seeds throughout so training runs are
     reproducible given the same data. Essential for debugging regressions.

Usage (called by retrain.yml):
    python training/train.py \
        --output-dir ./model_artifact_abc1234 \
        --model-version abc1234

Usage (local):
    python training/train.py --output-dir ./my_model --dry-run
"""

import argparse
import json
import logging

import random
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Fixed seeds for reproducibility
RANDOM_SEED = 42


def set_seeds(seed: int = RANDOM_SEED) -> None:
    """Set all random seeds for reproducible training."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def load_training_data(data_dir: Path) -> tuple[list[str], list[int]]:
    """
    Loads training data from training/data/.

    Expected format: one or more CSV files with columns:
        text,label
        "This product is great!",1
        "Terrible experience.",0

    Labels: 1 = POSITIVE, 0 = NEGATIVE

    Returns (texts, labels) as parallel lists.
    Falls back to a minimal synthetic dataset if no data files exist.
    This lets the CI pipeline run without real training data — the
    workflow tests the PIPELINE, not the model quality.
    """
    texts: list[str] = []
    labels: list[int] = []

    csv_files = list(data_dir.glob("*.csv"))

    if not csv_files:
        logger.warning(
            "No CSV files found in %s — using synthetic training data. "
            "This is only acceptable for pipeline validation, not real retraining.",
            data_dir,
        )
        # Minimal synthetic dataset — large enough to fine-tune the head
        synthetic = [
            ("This product is absolutely wonderful and exceeded my expectations.", 1),
            ("I love everything about this service, highly recommended!", 1),
            ("Outstanding quality, will definitely purchase again.", 1),
            ("Best purchase I have made in years, truly excellent.", 1),
            ("Fantastic experience from start to finish, five stars.", 1),
            ("Terrible product, broke after one day of use.", 0),
            ("Completely disappointed, would not recommend to anyone.", 0),
            ("Worst customer service I have ever experienced.", 0),
            ("Total waste of money, avoid at all costs.", 0),
            ("Absolutely dreadful quality, returned immediately.", 0),
        ] * 20  # Repeat to give the optimizer enough gradient signal
        texts, labels = zip(*synthetic)
        return list(texts), list(labels)

    logger.info("Found %d CSV file(s) in %s", len(csv_files), data_dir)

    for csv_file in csv_files:
        try:
            import csv

            with open(csv_file, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    text = row.get("text", "").strip()
                    label_raw = row.get("label", "").strip()
                    if not text or label_raw not in ("0", "1"):
                        continue
                    texts.append(text)
                    labels.append(int(label_raw))
            logger.info("Loaded %d examples from %s", len(texts), csv_file.name)
        except Exception as exc:
            logger.error("Failed to load %s: %s", csv_file, exc)

    if not texts:
        logger.error(
            "No valid training examples loaded — check CSV format (text,label)"
        )
        sys.exit(1)

    logger.info("Total training examples: %d", len(texts))
    return texts, labels


def train(
    output_dir: Path,
    model_version: str,
    dry_run: bool = False,
) -> dict:
    """
    Fine-tunes the classifier and saves the model artifact.

    Returns metrics dict from the training run (loss, accuracy on train split).
    The held-out evaluation against the test set is done in evaluate.py —
    not here. Separation of concerns: this script trains, that one evaluates.

    Why separate train and evaluate?
      The CI workflow can run them on different machines or at different times.
      More importantly: the promote-gate reads evaluate.py's output, not
      train.py's. Keeping them separate means the gate logic is independent
      of training implementation details.
    """
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback,
    )
    from sklearn.model_selection import train_test_split
    import torch
    from torch.utils.data import Dataset

    BASE_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

    logger.info(
        "Starting training run | version=%s | dry_run=%s", model_version, dry_run
    )

    # ---------------------------------------------------------------- #
    # Load data                                                         #
    # ---------------------------------------------------------------- #
    data_dir = Path(__file__).parent / "data"
    texts, labels = load_training_data(data_dir)

    # Hold out 20% for intra-training validation (not the final eval set)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=labels,
    )
    logger.info(
        "Split: %d train / %d validation",
        len(train_texts),
        len(val_texts),
    )

    # ---------------------------------------------------------------- #
    # Tokenizer & Model                                                 #
    # ---------------------------------------------------------------- #
    logger.info("Loading tokenizer and model: %s", BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1},
    )

    # Freeze all layers except the classifier head.
    # Why? Full fine-tuning on CPU in CI would take hours.
    # The base model already knows English semantics — we only need
    # to adjust the final classification weights for our data distribution.
    for name, param in model.named_parameters():
        if "classifier" not in name and "pre_classifier" not in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Trainable parameters: %d / %d (%.1f%%)",
        trainable,
        total,
        100 * trainable / total,
    )

    # ---------------------------------------------------------------- #
    # Dataset                                                           #
    # ---------------------------------------------------------------- #
    class SentimentDataset(Dataset):
        def __init__(self, texts: list[str], labels: list[int]) -> None:
            self.encodings = tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt",
            )
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, idx: int) -> dict:
            return {
                "input_ids": self.encodings["input_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx],
                "labels": self.labels[idx],
            }

    train_dataset = SentimentDataset(train_texts, train_labels)
    val_dataset = SentimentDataset(val_texts, val_labels)

    if dry_run:
        logger.info("Dry run — skipping actual training, writing stub artifact")
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        metrics = {"f1_score": 0.999, "accuracy": 0.999, "loss": 0.001, "dry_run": True}
        with open(output_dir / "train_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        return metrics

    # ---------------------------------------------------------------- #
    # Training arguments                                                #
    # ---------------------------------------------------------------- #
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-4,  # Higher LR OK since we're only training the head
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=RANDOM_SEED,
        no_cuda=True,  # Explicit CPU — no CUDA scan in CI
        report_to="none",  # Disable wandb/tensorboard in CI
        logging_steps=50,
        dataloader_num_workers=0,  # 0 workers in CI (no multiprocessing issues)
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # ---------------------------------------------------------------- #
    # Train                                                             #
    # ---------------------------------------------------------------- #
    logger.info("Starting training...")
    train_result = trainer.train()
    logger.info("Training complete. Loss: %.4f", train_result.training_loss)

    # ---------------------------------------------------------------- #
    # Save artifact                                                     #
    # ---------------------------------------------------------------- #
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Write training metadata alongside the weights
    train_metrics = {
        "training_loss": round(train_result.training_loss, 4),
        "train_runtime_seconds": round(train_result.metrics.get("train_runtime", 0), 1),
        "train_samples": len(train_texts),
        "val_samples": len(val_texts),
        "model_version": model_version,
        "base_model": BASE_MODEL,
    }

    with open(output_dir / "train_metrics.json", "w") as f:
        json.dump(train_metrics, f, indent=2)

    logger.info("Model artifact saved to %s", output_dir)
    logger.info("Training metrics: %s", train_metrics)

    return train_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune sentiment classifier")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save model weights and tokenizer",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="unknown",
        help="Git SHA or version tag for this training run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip actual training — write a stub artifact (for pipeline testing)",
    )
    args = parser.parse_args()

    set_seeds(RANDOM_SEED)
    metrics = train(
        output_dir=args.output_dir,
        model_version=args.model_version,
        dry_run=args.dry_run,
    )

    logger.info("Done. Training metrics: %s", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
