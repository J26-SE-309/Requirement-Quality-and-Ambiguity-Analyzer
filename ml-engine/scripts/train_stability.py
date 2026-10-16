
"""Task 3 prototype training. Run in Google Colab, never on local Windows.

Inputs: task3_train_filtered_annotations.csv and task3_val_filtered_annotations.csv.
Labels are provisional; metrics are diagnostics, NOT independent evaluation.
The held-out test file is deliberately not accepted or loaded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from quality_ml.stability_features import FEATURE_NAMES, extract_stability_features
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from xgboost import XGBClassifier

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

REQUIRED = {
    "pair_id",
    "original_text",
    "mutated_text",
    "stability_label",
}

LABEL_TO_INT = {
    "stable": 0,
    "unstable": 1,
}


def load_split(path: Path) -> pd.DataFrame:
    """Load and validate a provisional training or validation split."""

    frame = pd.read_csv(path, keep_default_na=False)

    missing = REQUIRED - set(frame.columns)

    if missing:
        raise ValueError(
            f"{path.name}: missing columns {sorted(missing)}"
        )

    if frame.empty or frame["pair_id"].duplicated().any():
        raise ValueError(
            f"{path.name}: empty or duplicate pair IDs"
        )

    if not frame["stability_label"].isin(LABEL_TO_INT).all():
        raise ValueError(
            f"{path.name}: unexpected or blank labels"
        )

    if (
        not frame["original_text"].astype(str).str.strip().all()
        or not frame["mutated_text"].astype(str).str.strip().all()
    ):
        raise ValueError(
            f"{path.name}: blank requirement text"
        )

    return frame


def make_features(
    frame: pd.DataFrame,
    encoder: SentenceTransformer,
    batch_size: int,
) -> np.ndarray:
    """Generate SBERT embeddings and extract stability features."""

    texts = pd.concat(
        [
            frame["original_text"],
            frame["mutated_text"],
        ],
        ignore_index=True,
    )

    unique_texts = list(dict.fromkeys(texts.tolist()))

    vectors = encoder.encode(
        unique_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    lookup = dict(
        zip(unique_texts, vectors, strict=True)
    )

    rows = [
        extract_stability_features(
            original,
            modified,
            lookup[original],
            lookup[modified],
        )
        for original, modified in zip(
            frame["original_text"],
            frame["mutated_text"],
            strict=True,
        )
    ]

    return np.asarray(
        [
            [row[name] for name in FEATURE_NAMES]
            for row in rows
        ],
        dtype=np.float32,
    )


def main() -> None:
    """Train and export the provisional Task 3 stability classifier."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--val",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    train = load_split(args.train)
    val = load_split(args.val)

    # Prevent the same pair from appearing in both splits.
    if set(train["pair_id"]) & set(val["pair_id"]):
        raise ValueError(
            "Train and validation pair IDs overlap"
        )

    # Prevent original requirements from crossing splits.
    if (
        "requirement_id" in train
        and "requirement_id" in val
        and set(train["requirement_id"]) & set(val["requirement_id"])
    ):
        raise ValueError(
            "Train and validation requirement IDs overlap"
        )

    print(
        f"Provisional data: train={len(train)}, val={len(val)}"
    )

    print(
        "WARNING: provisional validation metrics "
        "are NOT independent research results."
    )

    # Load pretrained SBERT in Google Colab.
    encoder = SentenceTransformer(MODEL_ID)

    # Generate embeddings and comparison features.
    x_train = make_features(
        train,
        encoder,
        args.batch_size,
    )

    x_val = make_features(
        val,
        encoder,
        args.batch_size,
    )

    # Convert text labels to numeric classes.
    y_train = train["stability_label"].map(
        LABEL_TO_INT
    ).to_numpy(dtype=np.int32)

    y_val = val["stability_label"].map(
        LABEL_TO_INT
    ).to_numpy(dtype=np.int32)

    if len(set(y_train)) != 2:
        raise ValueError(
            "Training split must contain both labels"
        )

    # Train XGBoost using extracted features.
    classifier = XGBClassifier(
        n_estimators=120,
        max_depth=2,
        learning_rate=0.05,
        subsample=1.0,
        colsample_bytree=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=2,
    )

    classifier.fit(
        x_train,
        y_train,
    )

    # Provisional validation only.
    predicted = classifier.predict(x_val)

    metrics = {
        "evaluation_status": "PROVISIONAL_DIAGNOSTIC_ONLY",
        "train_rows": len(train),
        "validation_rows": len(val),
        "validation_accuracy": float(
            accuracy_score(y_val, predicted)
        ),
        "validation_macro_f1": float(
            f1_score(
                y_val,
                predicted,
                average="macro",
                zero_division=0,
            )
        ),
        "confusion_matrix_labels_stable_unstable": (
            confusion_matrix(
                y_val,
                predicted,
                labels=[0, 1],
            ).tolist()
        ),
        "classification_report": classification_report(
            y_val,
            predicted,
            labels=[0, 1],
            target_names=[
                "stable",
                "unstable",
            ],
            zero_division=0,
            output_dict=True,
        ),
    }

    # Export artifacts.
    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        classifier,
        args.output / "stability_xgboost.joblib",
    )

    config = {
        "model_version": "task3-prototype-sbert-xgboost-v1",
        "embedding_model": MODEL_ID,
        "feature_names": list(FEATURE_NAMES),
        "label_to_int": LABEL_TO_INT,
        "decision_threshold": 0.5,
        "evaluation_status": "PROVISIONAL_DIAGNOSTIC_ONLY",
    }

    (args.output / "config.json").write_text(
        json.dumps(
            config,
            indent=2,
        ),
        encoding="utf-8",
    )

    (args.output / "provisional_metrics.json").write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                key: value
                for key, value in metrics.items()
                if key != "classification_report"
            },
            indent=2,
        )
    )

    print(
        "Artifacts:",
        args.output,
    )


if __name__ == "__main__":
    main()
