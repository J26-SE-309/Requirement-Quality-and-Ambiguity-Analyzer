"""Evaluate simple antecedent-ranking baselines against DAMIR."""

from collections import defaultdict

from quality_ml.ambiguity_checks import (
    find_candidate_antecedents,
    find_pronouns,
)
from quality_ml.damir import load_damir

DAMIR_PATH = (
    r"C:\Users\ASUS\Desktop\Synapse"
    r"\Datasets\requirement-quality\DAMIR.xlsx"
)


def normalize(text: str) -> str:
    """Normalize candidate text for comparison."""

    return " ".join(text.strip().lower().split())


def main() -> None:
    """Evaluate nearest and first-candidate baselines."""

    df = load_damir(DAMIR_PATH)

    labels_by_id = defaultdict(dict)

    for row in (
        df[
            [
                "Id",
                "Candidate Antecedent",
                "ResolvedAs",
            ]
        ]
        .drop_duplicates()
        .itertuples(index=False)
    ):
        occurrence_id = row[0]
        candidate = normalize(row[1])
        label = row[2]

        labels_by_id[occurrence_id][candidate] = label

    occurrence_rows = (
        df[
            [
                "Id",
                "Context",
                "Pronoun",
                "Position",
            ]
        ]
        .drop_duplicates(
            subset=["Id", "Pronoun", "Position"]
        )
    )

    evaluated = 0
    first_correct = 0
    nearest_correct = 0

    for row in occurrence_rows.itertuples(index=False):
        occurrence_id = row[0]
        context = row[1]
        pronoun_text = row[2]
        pronoun_position = int(row[3])

        pronouns = find_pronouns(context)

        matching_pronouns = [
            pronoun
            for pronoun in pronouns
            if pronoun["position"] == pronoun_position
            and pronoun["text"].lower() == pronoun_text.lower()
        ]

        if not matching_pronouns:
            continue

        candidates = find_candidate_antecedents(
            context,
            pronoun_position,
        )

        labels = labels_by_id.get(
            occurrence_id,
            {},
        )

        matched_candidates = [
            candidate
            for candidate in candidates
            if normalize(candidate["text"]) in labels
        ]

        if not matched_candidates:
            continue

        evaluated += 1

        # Candidate order is from earliest to latest occurrence
        # in the requirement text.
        first_candidate = matched_candidates[0]

        first_label = labels.get(
            normalize(first_candidate["text"])
        )

        if first_label == "correct":
            first_correct += 1

        # Find the candidate whose root is closest to the pronoun.
        nearest_candidate = min(
            matched_candidates,
            key=lambda candidate: abs(
                pronoun_position
                - candidate["root_position"]
            ),
        )

        nearest_label = labels.get(
            normalize(nearest_candidate["text"])
        )

        if nearest_label == "correct":
            nearest_correct += 1

    first_accuracy = (
        first_correct / evaluated
        if evaluated
        else 0.0
    )

    nearest_accuracy = (
        nearest_correct / evaluated
        if evaluated
        else 0.0
    )

    print("DAMIR antecedent-ranking baselines")
    print("==================================")
    print(f"Evaluated occurrences: {evaluated}")
    print()
    print(
        "First candidate correct: "
        f"{first_correct}"
    )
    print(
        "First candidate Top-1 accuracy: "
        f"{first_accuracy:.4%}"
    )
    print()
    print(
        "Nearest candidate correct: "
        f"{nearest_correct}"
    )
    print(
        "Nearest candidate Top-1 accuracy: "
        f"{nearest_accuracy:.4%}"
    )


if __name__ == "__main__":
    main()
