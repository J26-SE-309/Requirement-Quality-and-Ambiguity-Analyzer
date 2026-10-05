"""Tune the ambiguity decision threshold using the DAMIR validation split."""

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

GAP_THRESHOLDS = [
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.06,
    0.07,
    0.08,
    0.10,
    0.12,
    0.15,
]


def damir_to_binary_label(value: str) -> int:
    """Convert DAMIR AckUnack to binary ambiguity labels."""

    value = str(value).strip().lower()

    if value == "unambiguous":
        return 0

    if value in {"ack", "unack"}:
        return 1

    raise ValueError(
        f"Unknown DAMIR AckUnack value: {value}"
    )


def collect_validation_data(test_split):
    """Run the analyzer and collect validation evidence."""

    analyzer = Task2AmbiguityAnalyzer()

    records = []

    occurrence_groups = test_split.groupby(
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
                and item["pronoun"].lower()
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
                "similarity_gap": analysis[
                    "similarity_gap"
                ],
            }
        )

    return records


def predict_ambiguity(
    record: dict,
    gap_threshold: float,
) -> int:
    """Apply the threshold-based ambiguity decision rule."""

    if record["candidate_count"] < 2:
        return 0

    semantic_ambiguity = (
        record["similarity_gap"] < gap_threshold
    )

    syntactic_ambiguity = record["syntactic"]
    anaphoric_ambiguity = record["anaphoric"]

    evidence = sum(
        [
            semantic_ambiguity,
            syntactic_ambiguity,
            anaphoric_ambiguity,
        ]
    )

    return int(evidence >= 2)


def main():
    """Tune ambiguity threshold on validation only."""

    print("Loading DAMIR...")
    df = load_damir(DATA_PATH)

    print("Creating the fixed train/validation/test split...")
    _, validation, _ = split_damir(df)

    print()
    print("Validation split")
    print("================")
    print(f"Occurrences: {validation['Id'].nunique()}")

    records = collect_validation_data(validation)

    print(
        f"Evaluated occurrences: {len(records)}"
    )

    if not records:
        print("No validation occurrences could be evaluated.")
        return

    print()
    print("Threshold evaluation")
    print("====================")

    results = []

    for threshold in GAP_THRESHOLDS:

        y_true = [
            record["true_label"]
            for record in records
        ]

        y_pred = [
            predict_ambiguity(
                record,
                threshold,
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
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        results.append(result)

        print(
            f"threshold={threshold:.2f} | "
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
    print("Selected validation threshold")
    print("=============================")
    print(
        f"Semantic gap threshold: "
        f"{best['threshold']:.2f}"
    )
    print(f"Validation accuracy: {best['accuracy']:.4f}")
    print(f"Validation precision: {best['precision']:.4f}")
    print(f"Validation recall: {best['recall']:.4f}")
    print(f"Validation F1: {best['f1']:.4f}")

    print()
    print(
        "The test split was not used for threshold selection."
    )


if __name__ == "__main__":
    main()

