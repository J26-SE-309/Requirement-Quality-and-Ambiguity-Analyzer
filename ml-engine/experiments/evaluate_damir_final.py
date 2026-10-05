"""Final DAMIR candidate-ranking evaluation on the untouched test split."""

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

    return " ".join(text.strip().lower().split())


def build_labels(df):
    """Build candidate-level labels without losing conflicting annotations."""

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
    """Extract and score candidates for one DAMIR occurrence."""

    pronouns = find_pronouns(context)

    matching_pronouns = [
        pronoun
        for pronoun in pronouns
        if (
            pronoun["position"] == int(pronoun_position)
            and pronoun["text"].lower() == str(pronoun_text).lower()
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
        candidate_key = (
            str(occurrence_id),
            normalize_candidate(candidate["text"]),
        )

        if candidate_key in labels:
            matched_candidates.append(candidate)

    if not matched_candidates:
        return None

    interpretations = generate_interpretations(
        context,
        pronoun,
        matched_candidates,
    )

    if not interpretations:
        return None

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

        scored_candidates.append(
            {
                **candidate,
                "root_position": matching_extracted["root_position"],
                "dependency": matching_extracted["dependency"],
                "proximity_score": proximity_score,
                "syntactic_score": syntactic_score,
                "combined_score": combined_score,
            }
        )

    scored_candidates.sort(
        key=lambda candidate: candidate["combined_score"],
        reverse=True,
    )

    return {
        "occurrence_id": str(occurrence_id),
        "pronoun": pronoun,
        "candidates": scored_candidates,
    }


def main():
    """Evaluate the selected ranking configuration on the test split."""

    print("Loading DAMIR...")
    df = load_damir(DATA_PATH)

    print("Creating the fixed train/validation/test split...")
    _, _, test = split_damir(df)

    print()
    print("Test split")
    print("==========")
    print(f"Occurrences: {test['Id'].nunique()}")
    print(f"Candidate rows: {len(test)}")

    labels = build_labels(test)

    occurrence_groups = test.groupby(
        ["Id", "Context", "Pronoun", "Position"],
        sort=False,
    )

    evaluated_occurrences = 0
    occurrences_without_match = 0
    top1_correct = 0
    reciprocal_rank_sum = 0.0

    correct_candidates_considered = 0
    correct_candidates_ranked_first = 0

    for (
        occurrence_id,
        context,
        pronoun,
        position,
    ), _ in occurrence_groups:

        occurrence = prepare_occurrence(
            context=context,
            pronoun_text=pronoun,
            pronoun_position=position,
            occurrence_id=occurrence_id,
            labels=labels,
        )

        if occurrence is None:
            occurrences_without_match += 1
            continue

        scored_candidates = occurrence["candidates"]

        if not scored_candidates:
            occurrences_without_match += 1
            continue

        evaluated_occurrences += 1

        correct_ranks = []

        for rank, candidate in enumerate(
            scored_candidates,
            start=1,
        ):
            candidate_key = (
                str(occurrence_id),
                normalize_candidate(candidate["candidate"]),
            )

            if "correct" in labels[candidate_key]:
                correct_ranks.append(rank)

        correct_candidates_considered += len(correct_ranks)

        if correct_ranks:
            best_correct_rank = min(correct_ranks)

            if best_correct_rank == 1:
                top1_correct += 1
                correct_candidates_ranked_first += 1

            reciprocal_rank_sum += 1.0 / best_correct_rank

    print()
    print("Final DAMIR candidate-ranking results")
    print("=====================================")
    print(f"Semantic weight: {SEMANTIC_WEIGHT}")
    print(f"Proximity weight: {PROXIMITY_WEIGHT}")
    print(f"Syntactic weight: {SYNTACTIC_WEIGHT}")
    print()
    print(f"Evaluated occurrences: {evaluated_occurrences}")
    print(f"Occurrences without matched candidates: {occurrences_without_match}")

    if evaluated_occurrences:
        top1_accuracy = (
            top1_correct
            / evaluated_occurrences
        )

        mrr = (
            reciprocal_rank_sum
            / evaluated_occurrences
        )
    else:
        top1_accuracy = 0.0
        mrr = 0.0

    print()
    print(f"Top-1 accuracy: {top1_accuracy:.4f}")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print()
    print(
        "Correct candidates considered: "
        f"{correct_candidates_considered}"
    )
    print(
        "Correct candidates ranked first: "
        f"{correct_candidates_ranked_first}"
    )

    print()
    print(
        "This test split was not used during ranking-weight selection."
    )


if __name__ == "__main__":
    main()

