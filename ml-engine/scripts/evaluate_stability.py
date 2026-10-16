
"""Evaluate a saved Task 3 stability model on held-out test data.

Run in Google Colab after the evaluation procedure is finalized.

This script:
- Loads an existing XGBoost classifier and its configuration.
- Loads the corresponding frozen or fine-tuned SBERT encoder.
- Extracts the same features used during training.
- Evaluates predictions against held-out provisional labels.
- Saves metrics and per-pair predictions.

It does not train models, tune thresholds, or change saved artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from quality_ml.stability_features import FEATURE_NAMES
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from train_stability import load_split, make_features

LABEL_TO_INT = {
    "stable": 0,
    "unstable": 1,
}

INT_TO_LABEL = {
    0: "stable",
    1: "unstable",
}


def load_saved_model(model_dir: Path):
    """Load and verify the saved XGBoost classifier."""

    model_path = model_dir / "stability_xgboost.joblib"
    config_path = model_dir / "config.json"

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Saved XGBoost model not found: {model_path}"
        )

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Model configuration not found: {config_path}"
        )

    classifier = joblib.load(model_path)

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    if config.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError(
            "Saved feature names or order do not match training."
        )

    if classifier.n_features_in_ != len(FEATURE_NAMES):
        raise ValueError(
            "Saved model feature count does not match training."
        )

    if config.get("label_to_int") != LABEL_TO_INT:
        raise ValueError(
            "Saved label mapping does not match training."
        )

    if classifier.classes_.tolist() != [0, 1]:
        raise ValueError(
            "Saved model class order is incorrect."
        )

    if config.get("decision_threshold") != 0.5:
        raise ValueError(
            "Unexpected saved decision threshold."
        )

    return classifier, config


def evaluate_model(
    test_path: Path,
    model_dir: Path,
    output_dir: Path,
    encoder_override: str | None,
    batch_size: int,
) -> None:
    """Evaluate a previously trained classifier without retraining."""

    if not test_path.is_file():
        raise FileNotFoundError(
            f"Test dataset not found: {test_path}"
        )

    classifier, config = load_saved_model(model_dir)

    encoder_name = (
        encoder_override
        if encoder_override is not None
        else config["embedding_model"]
    )

    print("=" * 70)
    print("TASK 3 HELD-OUT TEST EVALUATION")
    print("=" * 70)

    print("Model directory:", model_dir)
    print("Encoder:", encoder_name)
    print("Feature count:", classifier.n_features_in_)
    print("Decision threshold:", config["decision_threshold"])

    # Load the held-out test dataset.
    test = load_split(test_path)

    print("\nTest pairs:", len(test))
    print("Loading SBERT encoder...")

    encoder = SentenceTransformer(encoder_name)

    # Reuse the exact training feature extraction implementation.
    print("\nExtracting test features...")

    x_test = make_features(
        test,
        encoder,
        batch_size=batch_size,
    )

    if x_test.shape != (len(test), len(FEATURE_NAMES)):
        raise ValueError(
            "Extracted feature matrix has an unexpected shape."
        )

    if not np.isfinite(x_test).all():
        raise ValueError(
            "Extracted features contain NaN or infinite values."
        )

    # Convert labels using the original training mapping.
    y_test = test["stability_label"].map(
        LABEL_TO_INT
    ).to_numpy(dtype=np.int32)

    # Predict using the saved model without fitting again.
    print("\nGenerating predictions...")

    predicted = classifier.predict(x_test).astype(np.int32)

    probabilities = classifier.predict_proba(x_test)[:, 1]

    # The saved XGBoost model uses a 0.5 decision threshold.
    threshold_predictions = (
        probabilities >= config["decision_threshold"]
    ).astype(np.int32)

    if not np.array_equal(predicted, threshold_predictions):
        raise ValueError(
            "Predictions do not match the saved decision threshold."
        )

    # Calculate final held-out test metrics.
    metrics = {
        "evaluation_status": "HELD_OUT_TEST_PROVISIONAL_LABELS",
        "label_status": "AI_ASSISTED_PROVISIONAL",
        "model_version": config["model_version"],
        "model_directory": str(model_dir),
        "encoder": encoder_name,
        "test_dataset": test_path.name,
        "test_rows": len(test),
        "feature_names": list(FEATURE_NAMES),
        "decision_threshold": config["decision_threshold"],
        "test_accuracy": float(
            accuracy_score(y_test, predicted)
        ),
        "test_macro_precision": float(
            precision_score(
                y_test,
                predicted,
                average="macro",
                zero_division=0,
            )
        ),
        "test_macro_recall": float(
            recall_score(
                y_test,
                predicted,
                average="macro",
                zero_division=0,
            )
        ),
        "test_macro_f1": float(
            f1_score(
                y_test,
                predicted,
                average="macro",
                zero_division=0,
            )
        ),
        "confusion_matrix_labels_stable_unstable": (
            confusion_matrix(
                y_test,
                predicted,
                labels=[0, 1],
            ).tolist()
        ),
        "classification_report": classification_report(
            y_test,
            predicted,
            labels=[0, 1],
            target_names=["stable", "unstable"],
            zero_division=0,
            output_dict=True,
        ),
    }

    # Save evaluation results separately from trained models.
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = output_dir / "test_metrics.json"

    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    # Save individual predictions for later error analysis.
    predictions_frame = test[
        ["pair_id", "stability_label"]
    ].copy()

    predictions_frame["predicted_label"] = [
        INT_TO_LABEL[int(value)]
        for value in predicted
    ]

    predictions_frame["unstable_probability"] = probabilities

    predictions_frame["correct"] = (
        predictions_frame["stability_label"]
        == predictions_frame["predicted_label"]
    )

    predictions_path = output_dir / "test_predictions.csv"

    predictions_frame.to_csv(
        predictions_path,
        index=False,
    )

    # Display the evaluation summary.
    print("\n" + "=" * 70)
    print("HELD-OUT TEST RESULTS")
    print("=" * 70)

    print("Test pairs:", metrics["test_rows"])
    print("Accuracy:", round(metrics["test_accuracy"], 4))
    print(
        "Macro precision:",
        round(metrics["test_macro_precision"], 4),
    )
    print(
        "Macro recall:",
        round(metrics["test_macro_recall"], 4),
    )
    print(
        "Macro F1:",
        round(metrics["test_macro_f1"], 4),
    )

    print("\nConfusion matrix [stable, unstable]:")
    print(
        np.array(
            metrics["confusion_matrix_labels_stable_unstable"]
        )
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predicted,
            labels=[0, 1],
            target_names=["stable", "unstable"],
            zero_division=0,
        )
    )

    print("\nSaved metrics:", metrics_path)
    print("Saved predictions:", predictions_path)

    print("\nIMPORTANT:")
    print(
        "These results use AI-assisted provisional labels. "
        "They do not establish independently human-verified accuracy."
    )


def main() -> None:
    """Parse arguments and run held-out evaluation."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a saved Task 3 XGBoost classifier "
            "without retraining."
        )
    )

    parser.add_argument(
        "--test",
        type=Path,
        required=True,
        help="Path to the held-out test CSV.",
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Directory containing the saved XGBoost model and config.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for evaluation results.",
    )

    parser.add_argument(
        "--encoder",
        type=str,
        default=None,
        help=(
            "Optional SBERT encoder path override. "
            "Use only to relocate the same saved encoder."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="SBERT inference batch size.",
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    evaluate_model(
        test_path=args.test,
        model_dir=args.model_dir,
        output_dir=args.output,
        encoder_override=args.encoder,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
