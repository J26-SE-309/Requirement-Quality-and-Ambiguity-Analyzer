"""Diagnose ambiguity-related features on the DAMIR validation split."""

from pathlib import Path

import pandas as pd
from quality_ml.ambiguity_checks import (
    find_candidate_antecedents,
    find_pronouns,
)
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


def normalize(text: str) -> str:
    """Normalize text for matching."""

    return " ".join(str(text).strip().lower().split())


def analyze_occurrence(
    occurrence_id,
    context,
    pronoun_text,
    position,
    ack_unack,
):
    """Extract ambiguity-related features for one occurrence."""

    pronouns = find_pronouns(context)

    matching = [
        pronoun
        for pronoun in pronouns
        if (
            pronoun["position"] == int(position)
            and pronoun["text"].lower()
            == str(pronoun_text).lower()
        )
    ]

    if not matching:
        return None

    pronoun = matching[0]

    candidates = find_candidate_antecedents(
        context,
        pronoun["position"],
    )

    if not candidates:
        return {
            "id": str(occurrence_id),
            "pronoun": pronoun["text"],
            "pronoun_lemma": pronoun["lemma"],
            "damir_label": ack_unack,
            "candidate_count": 0,
            "top_similarity": 0.0,
            "second_similarity": 0.0,
            "semantic_gap": 1.0,
            "top_proximity": 0.0,
            "second_proximity": 0.0,
            "proximity_gap": 0.0,
            "top_combined": 0.0,
            "second_combined": 0.0,
            "combined_gap": 0.0,
            "subject_candidates": 0,
            "object_candidates": 0,
            "conj_candidates": 0,
            "candidate_sentences": 0,
        }

    interpretations = generate_interpretations(
        context,
        pronoun,
        candidates,
    )

    ranked = rank_candidates(
        context,
        interpretations,
    )

    scored = []

    for ranked_candidate in ranked:
        candidate_text = ranked_candidate["candidate"]

        extracted = next(
            (
                candidate
                for candidate in candidates
                if normalize(candidate["text"])
                == normalize(candidate_text)
            ),
            None,
        )

        if extracted is None:
            continue

        proximity = calculate_proximity_score(
            pronoun["position"],
            extracted["root_position"],
        )

        syntactic = calculate_syntactic_score(
            extracted["dependency"],
        )

        combined = calculate_combined_score(
            semantic_score=ranked_candidate["similarity"],
            proximity_score=proximity,
            syntactic_score=syntactic,
            semantic_weight=SEMANTIC_WEIGHT,
            proximity_weight=PROXIMITY_WEIGHT,
            syntactic_weight=SYNTACTIC_WEIGHT,
        )

        scored.append(
            {
                "similarity": ranked_candidate["similarity"],
                "proximity": proximity,
                "combined": combined,
                "dependency": extracted["dependency"],
                "root_position": extracted["root_position"],
            }
        )

    scored.sort(
        key=lambda candidate: candidate["combined"],
        reverse=True,
    )

    top = scored[0] if scored else None
    second = scored[1] if len(scored) >= 2 else None

    candidate_sentences = set()

    for candidate in candidates:
        candidate_sentences.add(
            context[:candidate["start"]].count(".")
        )

    return {
        "id": str(occurrence_id),
        "pronoun": pronoun["text"],
        "pronoun_lemma": pronoun["lemma"],
        "damir_label": ack_unack,
        "candidate_count": len(candidates),
        "top_similarity": (
            top["similarity"]
            if top
            else 0.0
        ),
        "second_similarity": (
            second["similarity"]
            if second
            else 0.0
        ),
        "semantic_gap": (
            top["similarity"] - second["similarity"]
            if second
            else 1.0
        ),
        "top_proximity": (
            top["proximity"]
            if top
            else 0.0
        ),
        "second_proximity": (
            second["proximity"]
            if second
            else 0.0
        ),
        "proximity_gap": (
            top["proximity"] - second["proximity"]
            if second
            else 1.0
        ),
        "top_combined": (
            top["combined"]
            if top
            else 0.0
        ),
        "second_combined": (
            second["combined"]
            if second
            else 0.0
        ),
        "combined_gap": (
            top["combined"] - second["combined"]
            if second
            else 1.0
        ),
        "subject_candidates": sum(
            candidate["dependency"]
            in {"nsubj", "nsubjpass"}
            for candidate in candidates
        ),
        "object_candidates": sum(
            candidate["dependency"]
            in {"obj", "dobj", "pobj"}
            for candidate in candidates
        ),
        "conj_candidates": sum(
            candidate["dependency"] == "conj"
            for candidate in candidates
        ),
        "candidate_sentences": len(candidate_sentences),
    }


def main():
    """Run the validation feature diagnostic."""

    print("Loading DAMIR...")
    df = load_damir(DATA_PATH)

    print("Creating fixed train/validation/test split...")
    _, validation, _ = split_damir(df)

    print()
    print("Validation split")
    print("================")
    print(f"Occurrences: {validation['Id'].nunique()}")
    print(f"Candidate rows: {len(validation)}")

    groups = validation.groupby(
        ["Id", "Context", "Pronoun", "Position", "AckUnack"],
        sort=False,
    )

    rows = []

    print()
    print("Extracting features...")

    for (
        occurrence_id,
        context,
        pronoun,
        position,
        ack_unack,
    ), _ in groups:

        result = analyze_occurrence(
            occurrence_id=occurrence_id,
            context=context,
            pronoun_text=pronoun,
            position=position,
            ack_unack=ack_unack,
        )

        if result is not None:
            rows.append(result)

    features = pd.DataFrame(rows)

    print()
    print("Feature summary")
    print("================")

    numeric_columns = [
        "candidate_count",
        "top_similarity",
        "second_similarity",
        "semantic_gap",
        "top_proximity",
        "second_proximity",
        "proximity_gap",
        "top_combined",
        "second_combined",
        "combined_gap",
        "subject_candidates",
        "object_candidates",
        "conj_candidates",
        "candidate_sentences",
    ]

    summary = (
        features.groupby("damir_label")[numeric_columns]
        .agg(["mean", "median"])
        .round(4)
    )

    print(summary.to_string())

    print()
    print("Occurrence counts by DAMIR label")
    print("================================")

    print(
        features["damir_label"]
        .value_counts()
        .to_string()
    )

    print()
    print("Individual validation occurrences")
    print("=================================")

    display_columns = [
        "id",
        "pronoun",
        "damir_label",
        "candidate_count",
        "semantic_gap",
        "proximity_gap",
        "combined_gap",
        "subject_candidates",
        "object_candidates",
        "conj_candidates",
    ]

    print(
        features[
            display_columns
        ]
        .sort_values(
            ["damir_label", "combined_gap"]
        )
        .to_string(index=False)
    )

    print()
    print("Diagnostic complete.")
    print("The test split was not used.")


if __name__ == "__main__":
    main()
