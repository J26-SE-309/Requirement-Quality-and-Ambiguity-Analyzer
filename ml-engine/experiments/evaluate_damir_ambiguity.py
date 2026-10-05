"""Evaluate Task 2 ambiguity detection on the untouched DAMIR test split."""

from pathlib import Path

from quality_ml.damir import load_damir, split_damir
from quality_ml.task2 import Task2AmbiguityAnalyzer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

DATA_PATH = (
    Path(__file__).resolve().parents[3]
    / "Datasets"
    / "requirement-quality"
    / "DAMIR.xlsx"
)


def damir_to_binary_label(value: str) -> int:
    """Convert DAMIR occurrence labels to binary ambiguity labels."""

    value = str(value).strip().lower()

    if value == "unambiguous":
        return 0

    if value in {"ack", "unack"}:
        return 1

    raise ValueError(
        f"Unknown DAMIR AckUnack value: {value}"
    )


def main():
    """Evaluate binary ambiguity detection on the test split."""

    print("Loading DAMIR...")
    df = load_damir(DATA_PATH)

    print("Creating the fixed train/validation/test split...")
    _, _, test = split_damir(df)

    print()
    print("Test split")
    print("==========")
    print(f"Occurrences: {test['Id'].nunique()}")

    analyzer = Task2AmbiguityAnalyzer()

    y_true = []
    y_pred = []

    occurrence_groups = test.groupby(
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

        pronoun_results = [
            item
            for item in result["pronouns"]
            if (
                item["position"] == int(position)
                and item["pronoun"].lower()
                == str(pronoun).lower()
            )
        ]

        if not pronoun_results:
            continue

        prediction = pronoun_results[0]["is_ambiguous"]

        true_label = damir_to_binary_label(ack_unack)

        y_true.append(true_label)
        y_pred.append(int(prediction))

    print()
    print("Ambiguity detection results")
    print("============================")
    print(f"Evaluated occurrences: {len(y_true)}")

    if not y_true:
        print("No occurrences could be evaluated.")
        return

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")

    print()
    print("Confusion matrix")
    print("================")
    print(matrix)

    print()
    print("Classification report")
    print("=====================")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "unambiguous",
                "ambiguous",
            ],
            zero_division=0,
        )
    )

    print()
    print(
        "DAMIR mapping used for this evaluation:"
    )
    print(
        "Unambiguous -> 0 (unambiguous)"
    )
    print(
        "Ack / Unack -> 1 (ambiguous)"
    )

    print()
    print(
        "The test split was not used to tune the ranking weights."
    )


if __name__ == "__main__":
    main()
