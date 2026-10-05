"""Evaluate SBERT candidate ranking against DAMIR annotations."""

from collections import defaultdict

from quality_ml.ambiguity_checks import (
    find_candidate_antecedents,
    find_pronouns,
)
from quality_ml.damir import load_damir
from quality_ml.interpretations import generate_interpretations
from quality_ml.semantic_similarity import rank_candidates

DAMIR_PATH = (
    r"C:\Users\ASUS\Desktop\Synapse"
    r"\Datasets\requirement-quality\DAMIR.xlsx"
)


def normalize(text: str) -> str:
    """Normalize candidate text for comparison."""

    return " ".join(text.strip().lower().split())


def main() -> None:
    """Evaluate SBERT ranking against DAMIR."""

    df = load_damir(DAMIR_PATH)

    # Build candidate-level labels.
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

    evaluated_occurrences = 0
    top1_correct = 0
    reciprocal_rank_sum = 0.0

    total_correct_candidates = 0
    correct_candidates_ranked_first = 0

    for row in occurrence_rows.itertuples(index=False):
        occurrence_id = row[0]
        context = row[1]
        pronoun_text = row[2]
        position = int(row[3])

        pronouns = find_pronouns(context)

        matching_pronouns = [
            pronoun
            for pronoun in pronouns
            if pronoun["position"] == position
            and pronoun["text"].lower() == pronoun_text.lower()
        ]

        if not matching_pronouns:
            continue

        pronoun = matching_pronouns[0]

        extracted_candidates = find_candidate_antecedents(
            context,
            position,
        )

        if not extracted_candidates:
            continue

        damir_labels = labels_by_id.get(
            occurrence_id,
            {},
        )

        # Keep only candidates that have a DAMIR annotation.
        matched_candidates = [
            candidate
            for candidate in extracted_candidates
            if normalize(candidate["text"]) in damir_labels
        ]

        if not matched_candidates:
            continue

        interpretations = generate_interpretations(
            context,
            pronoun,
            matched_candidates,
        )

        ranked = rank_candidates(
            context,
            interpretations,
        )

        if not ranked:
            continue

        evaluated_occurrences += 1

        correct_ranks = []

        for rank, candidate in enumerate(ranked, start=1):
            candidate_name = normalize(
                candidate["candidate"]
            )

            label = damir_labels.get(candidate_name)

            if label == "correct":
                correct_ranks.append(rank)

        total_correct_candidates += len(correct_ranks)

        if correct_ranks:
            best_correct_rank = min(correct_ranks)

            reciprocal_rank_sum += (
                1.0 / best_correct_rank
            )

            if best_correct_rank == 1:
                top1_correct += 1

            if best_correct_rank == 1:
                correct_candidates_ranked_first += 1

    top1_accuracy = (
        top1_correct / evaluated_occurrences
        if evaluated_occurrences
        else 0.0
    )

    mrr = (
        reciprocal_rank_sum / evaluated_occurrences
        if evaluated_occurrences
        else 0.0
    )

    print("DAMIR SBERT candidate ranking")
    print("=============================")
    print(
        "Evaluated occurrences: "
        f"{evaluated_occurrences}"
    )
    print(
        "Occurrences with correct candidate ranked first: "
        f"{top1_correct}"
    )
    print(
        "Top-1 accuracy: "
        f"{top1_accuracy:.4%}"
    )
    print(
        "Mean Reciprocal Rank (MRR): "
        f"{mrr:.4f}"
    )
    print(
        "Correct candidates considered: "
        f"{total_correct_candidates}"
    )
    print(
        "Correct candidates ranked first: "
        f"{correct_candidates_ranked_first}"
    )


if __name__ == "__main__":
    main()
