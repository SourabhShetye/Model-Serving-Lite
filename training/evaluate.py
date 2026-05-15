"""
training/evaluate.py

Held-out evaluation script. Produces metrics.json for the promote-gate.

This is the script the promote-gate reads. Its output IS the gate.
That means the metrics it computes must be:
  1. Honest — evaluated on data the model has never seen
  2. Reproducible — same data + same model = same metrics, always
  3. Machine-readable — clean JSON, no human-formatted strings in numeric fields

Why a separate evaluation script instead of evaluating inside train.py?
  The training validation set is used for early stopping and checkpoint
  selection — the model HAS seen its loss, indirectly. It's not a true
  held-out set. The test set in evaluate.py is completely separate and
  touched only once: here. This is standard ML practice.

  Additionally, the CI workflow can call evaluate.py independently of
  train.py — useful for re-evaluating an existing artifact without
  retraining.

Usage (called by retrain.yml):
    python training/evaluate.py \
        --model-dir ./model_artifact \
        --output-file ./metrics.json

Usage (local, re-evaluate an existing model):
    python training/evaluate.py \
        --model-dir ./my_model \
        --output-file ./my_metrics.json \
        --verbose
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_test_data(data_dir: Path) -> tuple[list[str], list[int]]:
    """
    Loads test data from training/data/test_*.csv files.

    Convention: files prefixed with 'test_' are the held-out set.
    Files without the prefix are training data.

    Falls back to a synthetic test set if no test files exist.
    This ensures the CI pipeline always produces metrics.json,
    even in a repository with no real data yet.
    """
    test_files = list(data_dir.glob("test_*.csv"))

    if not test_files:
        logger.warning(
            "No test_*.csv files found — using synthetic test set. "
            "Metrics from synthetic data are not meaningful for production decisions."
        )
        # Synthetic test set — distinct from the synthetic training data
        return (
            [
                "Absolutely brilliant product, exceeded every expectation I had.",
                "I am very happy with this purchase, works perfectly.",
                "Good quality and fast shipping, very satisfied customer.",
                "Would not recommend this at all, very poor quality product.",
                "Extremely disappointed, this was a complete waste of money.",
                "This is amazing and I love using it every single day.",
                "Horrible experience, customer service was completely useless.",
                "Perfect in every way, could not ask for anything better.",
                "Dreadful product, fell apart within the first week of use.",
                "Exceptional value for money, far better than I expected.",
            ],
            [1, 1, 1, 0, 0, 1, 0, 1, 0, 1],
        )

    texts: list[str] = []
    labels: list[int] = []

    for test_file in test_files:
        import csv
        with open(test_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get("text", "").strip()
                label_raw = row.get("label", "").strip()
                if not text or label_raw not in ("0", "1"):
                    continue
                texts.append(text)
                labels.append(int(label_raw))
        logger.info("Loaded %d test examples from %s", len(texts), test_file.name)

    if not texts:
        logger.error("No valid test examples found — cannot evaluate")
        sys.exit(1)

    return texts, labels


def evaluate(
    model_dir: Path,
    output_file: Path,
    verbose: bool = False,
) -> dict:
    """
    Runs the fine-tuned model against the held-out test set.
    Writes metrics.json and returns the metrics dict.

    Metrics computed:
      - accuracy:   fraction of correct predictions
      - f1_score:   macro-averaged F1 (handles class imbalance)
      - precision:  macro-averaged precision
      - recall:     macro-averaged recall
      - n_samples:  test set size (the gate should sanity-check this)
      - per_class:  per-class breakdown (useful for debugging label skew)
    """
    from transformers import pipeline
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        classification_report,
        confusion_matrix,
    )

    logger.info("Loading model from %s", model_dir)

    if not model_dir.exists():
        logger.error("Model directory not found: %s", model_dir)
        sys.exit(1)

    # Load the fine-tuned pipeline from the saved artifact
    pipe = pipeline(
        "sentiment-analysis",
        model=str(model_dir),
        device=-1,          # CPU
        top_k=1,
    )

    # ---------------------------------------------------------------- #
    # Load test data                                                    #
    # ---------------------------------------------------------------- #
    data_dir = Path(__file__).parent / "data"
    texts, true_labels = load_test_data(data_dir)
    logger.info("Evaluating on %d test examples", len(texts))

    # ---------------------------------------------------------------- #
    # Run inference                                                     #
    # ---------------------------------------------------------------- #
    # Batch inference for efficiency — pipeline handles batching internally
    raw_predictions = pipe(texts, batch_size=16)

    # Map pipeline output labels to integers
    # Pipeline returns "POSITIVE"/"NEGATIVE" — map to 1/0 to match our labels
    LABEL_MAP = {"POSITIVE": 1, "NEGATIVE": 0}
    predicted_labels = [
        LABEL_MAP.get(pred[0]["label"], -1)
        for pred in raw_predictions
    ]

    if -1 in predicted_labels:
        unknown = [texts[i] for i, p in enumerate(predicted_labels) if p == -1]
        logger.error("Unknown labels returned for %d examples: %s", len(unknown), unknown[:3])
        sys.exit(1)

    # ---------------------------------------------------------------- #
    # Compute metrics                                                   #
    # ---------------------------------------------------------------- #
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average="macro", zero_division=0)
    precision = precision_score(true_labels, predicted_labels, average="macro", zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average="macro", zero_division=0)

    # Per-class breakdown — shows if the model is biased toward one class
    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=["NEGATIVE", "POSITIVE"],
        output_dict=True,
        zero_division=0,
    )

    conf_matrix = confusion_matrix(true_labels, predicted_labels).tolist()

    if verbose:
        logger.info("\n%s", classification_report(
            true_labels,
            predicted_labels,
            target_names=["NEGATIVE", "POSITIVE"],
            zero_division=0,
        ))
        logger.info("Confusion matrix:\n%s", conf_matrix)

    # ---------------------------------------------------------------- #
    # Build metrics output                                              #
    # ---------------------------------------------------------------- #
    # IMPORTANT: f1_score and accuracy at the top level are the values
    # the promote-gate reads. Do not rename or nest them.
    metrics = {
        "f1_score": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "n_samples": len(texts),
        "per_class": {
            "NEGATIVE": {
                "precision": round(report["NEGATIVE"]["precision"], 4),
                "recall": round(report["NEGATIVE"]["recall"], 4),
                "f1": round(report["NEGATIVE"]["f1-score"], 4),
                "support": int(report["NEGATIVE"]["support"]),
            },
            "POSITIVE": {
                "precision": round(report["POSITIVE"]["precision"], 4),
                "recall": round(report["POSITIVE"]["recall"], 4),
                "f1": round(report["POSITIVE"]["f1-score"], 4),
                "support": int(report["POSITIVE"]["support"]),
            },
        },
        "confusion_matrix": conf_matrix,
        "model_dir": str(model_dir),
    }

    # ---------------------------------------------------------------- #
    # Write metrics.json                                                #
    # ---------------------------------------------------------------- #
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Metrics written to %s", output_file)
    logger.info(
        "Results — F1: %.4f | Accuracy: %.4f | Precision: %.4f | Recall: %.4f | n=%d",
        f1, accuracy, precision, recall, len(texts),
    )

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned sentiment model on held-out test set"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to saved model artifact (output of train.py)",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Path to write metrics.json (read by the promote-gate)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full classification report and confusion matrix",
    )
    args = parser.parse_args()

    metrics = evaluate(
        model_dir=args.model_dir,
        output_file=args.output_file,
        verbose=args.verbose,
    )

    # Exit non-zero if the model fails the absolute floor check.
    # This makes the CI step red before the promote-gate even runs —
    # gives a more specific error message than "gate rejected".
    min_f1 = 0.80
    if metrics["f1_score"] < min_f1:
        logger.error(
            "Evaluation FAILED: F1=%.4f is below minimum acceptable threshold %.2f",
            metrics["f1_score"],
            min_f1,
        )
        sys.exit(1)

    logger.info(
        "Evaluation PASSED: F1=%.4f exceeds threshold %.2f",
        metrics["f1_score"],
        min_f1,
    )


if __name__ == "__main__":
    main()
