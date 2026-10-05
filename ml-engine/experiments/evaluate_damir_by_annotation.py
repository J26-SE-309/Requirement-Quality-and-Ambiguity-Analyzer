"""Evaluate DAMIR candidate ranking separately by annotation type."""

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
    """Normalize candidate text for annotation matching."""

    return " ".join(
        text.strip().lower().split()
    )


def build_labels(df):
    """Build candidate-level ResolvedAs labels."""

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
    ack_unack,
    labels,
):
    """Prepare one occurrence for candidate-ranking evaluation."""

    pronouns = find_pronouns(context)

    matching_pronouns = [
        pronoun
        for pronoun in pronouns
        if (
            pronoun["position"] == int(pronoun_position)
            and pronoun["text"].lower()
            == str(pronoun_text).lower()
        )
    ]

    if not matching_pronouns:
        return None

    pronoun = matching_pronouns[0]

    extracted_candidates = find_candidate_antecedents(
        context,
        pronoun["position"],
    )

    matched_candidates = []

    for candidate in extracted_candidates:
        key = (
            str(occurrence_id),
            normalize_candidate(candidate["text"]),
        )

        if key in labels:
            matched_candidates.append(candidate)

    if not matched_candidates:
        return None

    interpretations = generate_interpretations(
        context,
        pronoun,
        matched_candidates,
    )

    ranked = rank_candidates(
        context,
        interpretations,
    )

    scored_candidates = []

    for candidate in ranked:
        matching_extracted = next(
            (
                extracted
                for extracted in matched_candidates
                if normalize_candidate(extracted["text"])
                == normalize_candidate(candidate["candidate"])
            ),
            None,
        )

        if matching_extracted is None:
            continue

        proximity_score = calculate_proximity_score(
            pronoun["position"],
            matching_extracted["root_position"],
        )

        syntactic_score = calculate_syntactic_score(
            matching_extracted["dependency"],
        )

        combined_score = calculate_combined_score(
            semantic_score=candidate["similarity"],
            proximity_score=proximity_score,
            syntactic_score=syntactic_score,
            semantic_weight=SEMANTIC_WEIGHT,
            proximity_weight=PROXIMITY_WEIGHT,
            syntactic_weight=SYNTACTIC_WEIGHT,
        )

        candidate_key = (
            str(occurrence_id),
            normalize_candidate(candidate["candidate"]),
        )

        scored_candidates.append(
            {
                **candidate,
                "combined_score": combined_score,
                "resolved_as": labels[candidate_key],
            }
        )

    scored_candidates.sort(
        key=lambda candidate: candidate["combined_score"],
        reverse=True,
    )

    return {
        "occurrence_id": str(occurrence_id),
        "ack_unack": ack_unack,
        "candidates": scored_candidates,
    }


def evaluate_occurrences(occurrences):
    """Calculate Top-1 and MRR."""

    evaluated = 0
    top1_correct = 0
    reciprocal_rank_sum = 0.0

    for occurrence in occurrences:
        candidates = occurrence["candidates"]

        if not candidates:
            continue

        evaluated += 1

        correct_rank = None

        for rank, candidate in enumerate(
            candidates,
            start=1,
        ):
            if "correct" in candidate["resolved_as"]:
                correct_rank = rank
                break

        if correct_rank == 1:
            top1_correct += 1

        if correct_rank is not None:
            reciprocal_rank_sum += 1.0 / correct_rank

    if evaluated == 0:
        return {
            "evaluated": 0,
            "top1": 0.0,
            "mrr": 0.0,
        }

    return {
        "evaluated": evaluated,
        "top1": top1_correct / evaluated,
        "mrr": reciprocal_rank_sum / evaluated,
    }


def prepare_split(df):
    """Prepare all evaluable occurrences."""

    labels = build_labels(df)

    occurrences = []

    groups = df.groupby(
        [
            "Id",
            "Context",
            "Pronoun",
            "Position",
            "AckUnack",
        ],
        sort=False,
    )

    for (
        occurrence_id,
        context,
        pronoun,
        position,
        ack_unack,
    ), _ in groups:

        occurrence = prepare_occurrence(
            context=context,
            pronoun_text=pronoun,
            pronoun_position=position,
            occurrence_id=occurrence_id,
            ack_unack=ack_unack,
            labels=labels,
        )

        if occurrence is not None:
            occurrences.append(occurrence)

    return occurrences


def print_results(title, occurrences):
    """Print ranking metrics grouped by annotation type."""

    print()
    print(title)
    print("=" * len(title))

    overall = evaluate_occurrences(occurrences)

    print()
    print("Overall")
    print(f"Evaluated: {overall['evaluated']}")
    print(f"Top-1: {overall['top1']:.4f}")
    print(f"MRR: {overall['mrr']:.4f}")

    for label in [
        "Unambiguous",
        "Unack",
        "Ack",
    ]:
        subset = [
            occurrence
            for occurrence in occurrences
            if occurrence["ack_unack"] == label
        ]

        result = evaluate_occurrences(subset)

        print()
        print(label)
        print(f"Evaluated: {result['evaluated']}")
        print(f"Top-1: {result['top1']:.4f}")
        print(f"MRR: {result['mrr']:.4f}")


def main():
    """Run annotation-specific DAMIR evaluation."""

    print("Loading DAMIR...")
    df = load_damir(DATA_PATH)

    print("Creating fixed train/validation/test split...")
    _, validation, test = split_damir(df)

    print()
    print("Preparing validation split...")
    validation_occurrences = prepare_split(validation)

    print_results(
        "Validation results",
        validation_occurrences,
    )

    print()
    print("Preparing test split...")
    test_occurrences = prepare_split(test)

    print_results(
        "Test results",
        test_occurrences,
    )

    print()
    print("Fixed ranking weights")
    print("=====================")
    print(f"Semantic: {SEMANTIC_WEIGHT}")
    print(f"Proximity: {PROXIMITY_WEIGHT}")
    print(f"Syntactic: {SYNTACTIC_WEIGHT}")

    print()
    print(
        "The test split was not used for selecting "
        "the ranking weights."
    )


if __name__ == "__main__":
    main()
