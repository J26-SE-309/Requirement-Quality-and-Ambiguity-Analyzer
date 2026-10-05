"""Tune candidate-ranking weights using the DAMIR validation split."""

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


WEIGHT_CONFIGURATIONS = [
    (0.7, 0.2, 0.1),
    (0.6, 0.3, 0.1),
    (0.6, 0.2, 0.2),
    (0.5, 0.4, 0.1),
    (0.5, 0.3, 0.2),
    (0.5, 0.2, 0.3),
    (0.4, 0.5, 0.1),
    (0.4, 0.4, 0.2),
    (0.4, 0.3, 0.3),
    (0.3, 0.5, 0.2),
    (0.3, 0.4, 0.3),
    (0.3, 0.3, 0.4),
]


def normalize_candidate(text: str) -> str:
    """Normalize candidate text for annotation matching."""

    return " ".join(text.strip().lower().split())


def build_labels(df):
    """Build candidate-level label sets without losing conflicting labels."""

    labels = defaultdict(set)

    for _, row in df.iterrows():
        key = (
            str(row["Id"]),
            normalize_candidate(row["Candidate Antecedent"]),
        )

        labels[key].add(str(row["ResolvedAs"]).strip().lower())

    return labels


def prepare_occurrence(context, pronoun_text, pronoun_position, occurrence_id, labels):
    """Prepare matched DAMIR candidates and their semantic ranking scores."""

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
        candidate_text = candidate["candidate"]

        matching_extracted = next(
            (
                extracted
                for extracted in matched_candidates
                if normalize_candidate(extracted["text"])
                == normalize_candidate(candidate_text)
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

        scored_candidates.append(
            {
                **candidate,
                "root_position": matching_extracted["root_position"],
                "dependency": matching_extracted["dependency"],
                "proximity_score": proximity_score,
                "syntactic_score": syntactic_score,
            }
        )

    return {
        "occurrence_id": str(occurrence_id),
        "pronoun": pronoun,
        "candidates": scored_candidates,
    }


def evaluate_prepared_occurrences(
    prepared_occurrences,
    semantic_weight,
    proximity_weight,
    syntactic_weight,
):
    """Evaluate one ranking-weight configuration."""

    top1_correct = 0
    reciprocal_rank_sum = 0.0
    evaluated_occurrences = 0

    for occurrence in prepared_occurrences:
        occurrence_id = occurrence["occurrence_id"]

        scored = []

        for candidate in occurrence["candidates"]:
            combined_score = calculate_combined_score(
                semantic_score=candidate["similarity"],
                proximity_score=candidate["proximity_score"],
                syntactic_score=candidate["syntactic_score"],
                semantic_weight=semantic_weight,
                proximity_weight=proximity_weight,
                syntactic_weight=syntactic_weight,
            )

            scored.append(
                {
                    **candidate,
                    "combined_score": combined_score,
                }
            )

        if not scored:
            continue

        scored.sort(
            key=lambda candidate: candidate["combined_score"],
            reverse=True,
        )

        evaluated_occurrences += 1

        correct_rank = None

        for rank, candidate in enumerate(scored, start=1):
            candidate_key = (
                occurrence_id,
                normalize_candidate(candidate["candidate"]),
            )

            if "correct" in occurrence["labels"].get(
                candidate_key,
                set(),
            ):
                correct_rank = rank
                break

        if correct_rank == 1:
            top1_correct += 1

        if correct_rank is not None:
            reciprocal_rank_sum += 1.0 / correct_rank

    if evaluated_occurrences == 0:
        return {
            "top1": 0.0,
            "mrr": 0.0,
            "occurrences": 0,
        }

    return {
        "top1": top1_correct / evaluated_occurrences,
        "mrr": reciprocal_rank_sum / evaluated_occurrences,
        "occurrences": evaluated_occurrences,
    }


def prepare_split(df):
    """Prepare all evaluable occurrences in one DAMIR split."""

    labels = build_labels(df)

    occurrence_groups = df.groupby(
        ["Id", "Context", "Pronoun", "Position"],
        sort=False,
    )

    prepared = []

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
            continue

        occurrence["labels"] = labels
        prepared.append(occurrence)

    return prepared


def main():
    """Run ranking-weight evaluation on the DAMIR validation split."""

    print("Loading DAMIR...")
    df = load_damir(DATA_PATH)

    print("Creating train/validation/test split...")
    train, validation, test = split_damir(df)

    print()
    print("DAMIR split sizes")
    print("=================")
    print(f"Train occurrences: {train['Id'].nunique()}")
    print(f"Validation occurrences: {validation['Id'].nunique()}")
    print(f"Test occurrences: {test['Id'].nunique()}")

    print()
    print("Preparing validation occurrences...")
    validation_occurrences = prepare_split(validation)

    print(
        "Validation occurrences with matched candidates: "
        f"{len(validation_occurrences)}"
    )

    print()
    print("Evaluating ranking configurations...")
    print()

    results = []

    for (
        semantic_weight,
        proximity_weight,
        syntactic_weight,
    ) in WEIGHT_CONFIGURATIONS:

        validation_result = evaluate_prepared_occurrences(
            validation_occurrences,
            semantic_weight,
            proximity_weight,
            syntactic_weight,
        )

        result = {
            "semantic": semantic_weight,
            "proximity": proximity_weight,
            "syntactic": syntactic_weight,
            "top1": validation_result["top1"],
            "mrr": validation_result["mrr"],
            "occurrences": validation_result["occurrences"],
        }

        results.append(result)

        print(
            f"semantic={semantic_weight:.1f}, "
            f"proximity={proximity_weight:.1f}, "
            f"syntactic={syntactic_weight:.1f} | "
            f"Top-1={validation_result['top1']:.4f} | "
            f"MRR={validation_result['mrr']:.4f}"
        )

    results.sort(
        key=lambda result: (
            result["mrr"],
            result["top1"],
        ),
        reverse=True,
    )

    best = results[0]

    print()
    print("Best validation configuration")
    print("=============================")
    print(f"Semantic weight: {best['semantic']:.1f}")
    print(f"Proximity weight: {best['proximity']:.1f}")
    print(f"Syntactic weight: {best['syntactic']:.1f}")
    print(f"Validation Top-1: {best['top1']:.4f}")
    print(f"Validation MRR: {best['mrr']:.4f}")
    print(f"Occurrences evaluated: {best['occurrences']}")

    print()
    print(
        "The test split was intentionally not used for selecting "
        "the ranking configuration."
    )


if __name__ == "__main__":
    main()
