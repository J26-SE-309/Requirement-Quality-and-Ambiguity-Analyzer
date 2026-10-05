"""Evaluate spaCy candidate extraction against DAMIR annotations."""

from collections import Counter, defaultdict

from quality_ml.ambiguity_checks import (
    find_candidate_antecedents,
    find_pronouns,
)
from quality_ml.damir import (
    candidate_annotations,
    load_damir,
)

DAMIR_PATH = (
    r"C:\Users\ASUS\Desktop\Synapse"
    r"\Datasets\requirement-quality\DAMIR.xlsx"
)


def normalize(text: str) -> str:
    """Normalize candidate text for exact comparison."""

    return " ".join(text.strip().lower().split())


def main() -> None:
    """Evaluate candidate extraction coverage."""

    df = load_damir(DAMIR_PATH)

    annotations = candidate_annotations(df)

    # Keep candidate annotations grouped by occurrence.
    annotated_by_id = defaultdict(set)

    for row in annotations.itertuples(index=False):
        occurrence_id = row[0]
        candidate_text = row[1]

        annotated_by_id[occurrence_id].add(
            normalize(candidate_text)
        )

    # Keep candidate labels grouped by occurrence and candidate.
    labels_by_id = defaultdict(dict)

    for row in annotations.itertuples(index=False):
        occurrence_id = row[0]
        candidate_text = normalize(row[1])
        label = row[2]

        labels_by_id[occurrence_id][candidate_text] = label

    # Count the actual unique candidate pairs.
    total_unique_candidates = len(
        df[
            [
                "Id",
                "Candidate Antecedent",
            ]
        ].drop_duplicates()
    )

    matched_candidates = 0

    occurrences_with_match = 0
    occurrences_without_match = 0

    coverage_by_label = Counter()
    matched_by_label = Counter()

    occurrence_rows = (
        df[
            [
                "Id",
                "Context",
                "Pronoun",
                "Position",
                "AckUnack",
            ]
        ]
        .drop_duplicates(
            subset=["Id", "Pronoun", "Position"]
        )
    )

    for row in occurrence_rows.itertuples(index=False):
        occurrence_id = row[0]
        context = row[1]
        pronoun = row[2]
        position = int(row[3])

        pronouns = find_pronouns(context)

        matching_pronouns = [
            item
            for item in pronouns
            if item["position"] == position
            and item["text"].lower() == pronoun.lower()
        ]

        if not matching_pronouns:
            occurrences_without_match += 1
            continue

        extracted = find_candidate_antecedents(
            context,
            position,
        )

        extracted_texts = {
            normalize(candidate["text"])
            for candidate in extracted
        }

        annotated = annotated_by_id.get(
            occurrence_id,
            set(),
        )

        matched = annotated & extracted_texts

        matched_candidates += len(matched)

        if matched:
            occurrences_with_match += 1
        else:
            occurrences_without_match += 1

        labels = labels_by_id.get(
            occurrence_id,
            {},
        )

        for candidate_text, label in labels.items():
            coverage_by_label[label] += 1

            if candidate_text in extracted_texts:
                matched_by_label[label] += 1

    coverage = (
        matched_candidates / total_unique_candidates
        if total_unique_candidates
        else 0.0
    )

    print("DAMIR candidate coverage")
    print("========================")
    print(
        "Unique candidate pairs: "
        f"{total_unique_candidates}"
    )
    print(f"Matched candidates: {matched_candidates}")
    print(f"Coverage: {coverage:.4%}")
    print()
    print("Occurrence coverage")
    print("===================")
    print(
        "Occurrences with at least one match: "
        f"{occurrences_with_match}"
    )
    print(
        "Occurrences without a match: "
        f"{occurrences_without_match}"
    )
    print()
    print("Coverage by ResolvedAs")
    print("======================")

    for label in sorted(coverage_by_label):
        total = coverage_by_label[label]
        matched = matched_by_label[label]

        label_coverage = (
            matched / total
            if total
            else 0.0
        )

        print(
            f"{label}: "
            f"{matched}/{total} "
            f"({label_coverage:.4%})"
        )


if __name__ == "__main__":
    main()
