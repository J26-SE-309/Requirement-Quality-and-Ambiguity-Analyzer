
"""Tune EXP-04 ambiguity thresholds using the DAMIR validation split."""

from pathlib import Path

from quality_ml.damir import load_damir, split_damir
from quality_ml.task2 import Task2AmbiguityAnalyzer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

DATA_PATH = (
    Path(__file__).resolve().parents[3]
    / "Datasets"
    / "requirement-quality"
    / "DAMIR.xlsx"
)

# Candidate thresholds for EXP-04 top-two cosine similarity margin.
GAP_THRESHOLDS = [
    0.005,
    0.01,
    0.015,
    0.02,
    0.03,
    0.04,
    0.05,
    0.075,
    0.10,
    0.15,
    0.20,
    0.30,
]

# Candidate thresholds for normalized entropy.
ENTROPY_THRESHOLDS = [
    0.0,
    0.70,
    0.80,
    0.85,
    0.90,
    0.95,
    0.98,
    0.99,
]


def damir_to_binary_label(value: str) -> int:
    """Convert DAMIR AckUnack labels to binary ambiguity labels."""

    value = str(value).strip().lower()

    if value == "unambiguous":
        return 0

    if value in {"ack", "unack"}:
        return 1

    raise ValueError(
        f"Unknown DAMIR AckUnack value: {value}"
    )


def collect_validation_data(validation_split):
    """Collect EXP-04 evidence for labeled validation occurrences."""

    analyzer = Task2AmbiguityAnalyzer()
    records = []

    occurrence_groups = validation_split.groupby(
        ["Id", "Context", "Pronoun", "Position", "AckUnack"],
        sort=False,
    )

    for (
        occurrence_id,
        context,
        pronoun,
        position,
        ack_unack,
    ), _ in occurrence_groups:

        result = analyzer.analyze(context)

        matching = [
            item
            for item in result["pronouns"]
            if (
                item["position"] == int(position)
                and item["reference"].lower()
                == str(pronoun).lower()
            )
        ]

        if not matching:
            continue

        analysis = matching[0]

        records.append(
            {
                "id": str(occurrence_id),
                "true_label": damir_to_binary_label(
                    ack_unack
                ),
                "candidate_count": analysis[
                    "candidate_count"
                ],
                "syntactic": analysis[
                    "signals"
                ]["syntactic"],
                "anaphoric": analysis[
                    "signals"
                ]["anaphoric"],
                "competition_score": analysis[
                    "competition_score"
                ],
                "top2_margin": analysis[
                    "top2_margin"
                ],
                "normalized_entropy": analysis[
                    "normalized_entropy"
                ],
                "similarity_gap": analysis[
                    "similarity_gap"
                ],
            }
        )

    return records


def predict_ambiguity(
    record: dict,
    gap_threshold: float,
    entropy_threshold: float = 0.0,
) -> int:
    """Predict ambiguity using EXP-04 competition evidence."""

    margin = record["top2_margin"]
    entropy = record["normalized_entropy"]

    return int(
        record["candidate_count"] >= 2
        and margin is not None
        and entropy is not None
        and margin <= gap_threshold
        and entropy >= entropy_threshold
    )


def main():
    """Tune margin and entropy thresholds on DAMIR validation only."""

    print("Loading DAMIR...")
    df = load_damir(DATA_PATH)

    print("Creating the fixed train/validation/test split...")
    _, validation, _ = split_damir(df)

    print()
    print("Validation split")
    print("================")
    print(f"Occurrences: {validation['Id'].nunique()}")

    records = collect_validation_data(validation)

    print(f"Evaluated occurrences: {len(records)}")

    if not records:
        print("No validation occurrences could be evaluated.")
        return

    print()
    print("Threshold evaluation")
    print("====================")

    y_true = [
        record["true_label"]
        for record in records
    ]

    results = []

    for threshold in GAP_THRESHOLDS:
        for entropy_threshold in ENTROPY_THRESHOLDS:

            y_pred = [
                predict_ambiguity(
                    record,
                    threshold,
                    entropy_threshold,
                )
                for record in records
            ]

            accuracy = accuracy_score(
                y_true,
                y_pred,
            )

            precision = precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )

            recall = recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )

            f1 = f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )

            result = {
                "threshold": threshold,
                "entropy_threshold": entropy_threshold,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

            results.append(result)

            print(
                f"margin={threshold:.3f} | "
                f"entropy={entropy_threshold:.2f} | "
                f"accuracy={accuracy:.4f} | "
                f"precision={precision:.4f} | "
                f"recall={recall:.4f} | "
                f"F1={f1:.4f}"
            )

    results.sort(
        key=lambda result: (
            result["f1"],
            result["accuracy"],
        ),
        reverse=True,
    )

    best = results[0]

    print()
    print("Selected validation thresholds")
    print("==============================")
    print(
        f"EXP-04 margin threshold: "
        f"{best['threshold']:.3f}"
    )
    print(
        f"Normalized entropy threshold: "
        f"{best['entropy_threshold']:.2f}"
    )
    print(
        f"Validation accuracy: "
        f"{best['accuracy']:.4f}"
    )
    print(
        f"Validation precision: "
        f"{best['precision']:.4f}"
    )
    print(
        f"Validation recall: "
        f"{best['recall']:.4f}"
    )
    print(
        f"Validation F1: "
        f"{best['f1']:.4f}"
    )

    print()
    print(
        "The test split was not used for threshold selection."
    )


if __name__ == "__main__":
    main()
