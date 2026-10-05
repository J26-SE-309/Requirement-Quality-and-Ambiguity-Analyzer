"""Inspect held-out DAMIR Unack ranking failures."""

from collections import defaultdict
from pathlib import Path

from quality_ml.ambiguity_checks import find_candidate_antecedents, find_pronouns
from quality_ml.damir import load_damir, split_damir
from quality_ml.interpretations import generate_interpretations
from quality_ml.ranking import (
    calculate_combined_score,
    calculate_proximity_score,
    calculate_syntactic_score,
)
from quality_ml.semantic_similarity import rank_candidates

DATA_PATH = (
    Path(__file__).resolve().parents[3]
    / "Datasets"
    / "requirement-quality"
    / "DAMIR.xlsx"
)

SEMANTIC_WEIGHT = 0.7
PROXIMITY_WEIGHT = 0.2
SYNTACTIC_WEIGHT = 0.1


def normalize_candidate(text: str) -> str:
    return " ".join(text.strip().lower().split())


def build_labels(df):
    labels = defaultdict(set)

    for _, row in df.iterrows():
        key = (
            str(row["Id"]),
            normalize_candidate(row["Candidate Antecedent"]),
        )

        labels[key].add(
            str(row["ResolvedAs"]).strip().lower()
        )

    return labels


def prepare_occurrence(
    context,
    pronoun_text,
    pronoun_position,
    occurrence_id,
    labels,
):
    pronouns = find_pronouns(context)

    matching = [
        pronoun
        for pronoun in pronouns
        if (
            pronoun["position"] == int(pronoun_position)
            and pronoun["text"].lower()
            == str(pronoun_text).lower()
        )
    ]

    if not matching:
        return None

    pronoun = matching[0]

    extracted = find_candidate_antecedents(
        context,
        pronoun["position"],
    )

    matched = []

    for candidate in extracted:
        key = (
            str(occurrence_id),
            normalize_candidate(candidate["text"]),
        )

        if key in labels:
            matched.append(candidate)

    if not matched:
        return None

    interpretations = generate_interpretations(
        context,
        pronoun,
        matched,
    )

    ranked = rank_candidates(
        context,
        interpretations,
    )

    results = []

    for candidate in ranked:
        extracted_candidate = next(
            (
                item
                for item in matched
                if normalize_candidate(item["text"])
                == normalize_candidate(candidate["candidate"])
            ),
            None,
        )

        if extracted_candidate is None:
            continue

        proximity = calculate_proximity_score(
            pronoun["position"],
            extracted_candidate["root_position"],
        )

        syntactic = calculate_syntactic_score(
            extracted_candidate["dependency"],
        )

        combined = calculate_combined_score(
            semantic_score=candidate["similarity"],
            proximity_score=proximity,
            syntactic_score=syntactic,
            semantic_weight=SEMANTIC_WEIGHT,
            proximity_weight=PROXIMITY_WEIGHT,
            syntactic_weight=SYNTACTIC_WEIGHT,
        )

        key = (
            str(occurrence_id),
            normalize_candidate(candidate["candidate"]),
        )

        results.append(
            {
                "candidate": candidate["candidate"],
                "similarity": candidate["similarity"],
                "proximity": proximity,
                "syntactic": syntactic,
                "combined": combined,
                "labels": labels[key],
            }
        )

    results.sort(
        key=lambda item: item["combined"],
        reverse=True,
    )

    return {
        "id": str(occurrence_id),
        "pronoun": pronoun_text,
        "position": pronoun_position,
        "context": context,
        "candidates": results,
    }


def main():
    print("Loading DAMIR...")
    df = load_damir(DATA_PATH)

    print("Creating fixed train/validation/test split...")
    _, _, test = split_damir(df)

    labels = build_labels(test)

    failures = []

    groups = test.groupby(
        ["Id", "Context", "Pronoun", "Position", "AckUnack"],
        sort=False,
    )

    for (
        occurrence_id,
        context,
        pronoun,
        position,
        ack_unack,
    ), _ in groups:

        if ack_unack != "Unack":
            continue

        occurrence = prepare_occurrence(
            context=context,
            pronoun_text=pronoun,
            pronoun_position=position,
            occurrence_id=occurrence_id,
            labels=labels,
        )

        if occurrence is None:
            continue

        candidates = occurrence["candidates"]

        correct = [
            candidate
            for candidate in candidates
            if "correct" in candidate["labels"]
        ]

        if not correct:
            continue

        if "correct" not in candidates[0]["labels"]:
            failures.append(occurrence)

    print()
    print("Unack ranking failures")
    print("======================")
    print(f"Failures with a known correct candidate: {len(failures)}")

    for occurrence in failures:
        print()
        print("=" * 100)
        print(f"ID: {occurrence['id']}")
        print(f"Pronoun: {occurrence['pronoun']}")
        print(f"Position: {occurrence['position']}")

        print()
        print("CONTEXT")
        print("-" * 100)
        print(occurrence["context"])

        print()
        print("RANKED CANDIDATES")
        print("-" * 100)

        for rank, candidate in enumerate(
            occurrence["candidates"],
            start=1,
        ):
            label = ",".join(sorted(candidate["labels"]))

            print(
                f"{rank:2d}. "
                f"{candidate['candidate']}"
                f" | combined={candidate['combined']:.4f}"
                f" | semantic={candidate['similarity']:.4f}"
                f" | proximity={candidate['proximity']:.4f}"
                f" | syntactic={candidate['syntactic']:.4f}"
                f" | ResolvedAs={label}"
            )

    print()
    print("=" * 100)
    print("Inspection complete.")
    print("Only held-out test Unack cases were inspected.")
    print("No model or production code was changed.")


if __name__ == "__main__":
    main()
