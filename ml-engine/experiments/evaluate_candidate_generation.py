"""Evaluate candidate-generation strategies on the DAMIR validation split."""

from pathlib import Path

from quality_ml.ambiguity_checks import find_candidate_antecedents, find_pronouns
from quality_ml.damir import load_damir, split_damir

DATA_PATH = (
    Path(__file__).resolve().parents[3]
    / "Datasets"
    / "requirement-quality"
    / "DAMIR.xlsx"
)


def normalize(text: str) -> str:
    """Normalize text for DAMIR candidate matching."""

    return " ".join(str(text).strip().lower().split())


def is_nested(candidate, other) -> bool:
    """Return whether candidate is completely contained inside another span."""

    return (
        other["start"] <= candidate["start"]
        and candidate["end"] <= other["end"]
        and (
            other["start"] < candidate["start"]
            or candidate["end"] < other["end"]
        )
    )


def remove_nested_candidates(candidates):
    """Remove noun chunks fully contained inside a larger candidate."""

    filtered = []

    for candidate in candidates:
        nested = any(
            is_nested(candidate, other)
            for other in candidates
            if other is not candidate
        )

        if not nested:
            filtered.append(candidate)

    return filtered


def remove_generated_conjunct_fragments(candidates):
    """Remove candidates whose dependency is conj."""

    return [
        candidate
        for candidate in candidates
        if candidate["dependency"] != "conj"
    ]


def get_strategies(candidates):
    """Return candidate sets for each ablation strategy."""

    nested_removed = remove_nested_candidates(candidates)

    conjuncts_removed = remove_generated_conjunct_fragments(candidates)

    combined = remove_generated_conjunct_fragments(
        nested_removed
    )

    return {
        "A_current": candidates,
        "B_no_nested": nested_removed,
        "C_no_conj": conjuncts_removed,
        "D_no_nested_no_conj": combined,
    }


def evaluate_strategy(validation):
    """Evaluate all candidate-generation strategies."""

    stats = {
        name: {
            "occurrences": 0,
            "occurrences_with_match": 0,
            "occurrences_with_correct": 0,
            "candidate_total": 0,
            "annotated_total": 0,
            "matched_total": 0,
            "correct_total": 0,
            "correct_matched": 0,
        }
        for name in [
            "A_current",
            "B_no_nested",
            "C_no_conj",
            "D_no_nested_no_conj",
        ]
    }

    groups = validation.groupby(
        ["Id", "Context", "Pronoun", "Position"],
        sort=False,
    )

    for (
        occurrence_id,
        context,
        pronoun_text,
        position,
    ), group in groups:

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
            continue

        pronoun = matching[0]

        candidates = find_candidate_antecedents(
            context,
            pronoun["position"],
        )

        annotated = {
            normalize(row["Candidate Antecedent"])
            for _, row in group.iterrows()
        }

        correct = {
            normalize(row["Candidate Antecedent"])
            for _, row in group.iterrows()
            if normalize(row["ResolvedAs"]) == "correct"
        }

        strategies = get_strategies(candidates)

        for name, strategy_candidates in strategies.items():
            strategy_texts = {
                normalize(candidate["text"])
                for candidate in strategy_candidates
            }

            matched = annotated & strategy_texts
            correct_matched = correct & strategy_texts

            stats[name]["occurrences"] += 1
            stats[name]["candidate_total"] += len(strategy_candidates)
            stats[name]["annotated_total"] += len(annotated)
            stats[name]["matched_total"] += len(matched)
            stats[name]["correct_total"] += len(correct)
            stats[name]["correct_matched"] += len(correct_matched)

            if matched:
                stats[name]["occurrences_with_match"] += 1

            if correct_matched:
                stats[name]["occurrences_with_correct"] += 1

    return stats


def print_results(stats):
    """Print the candidate-generation comparison."""

    print()
    print("Candidate-generation ablation")
    print("==============================")
    print()

    for name, result in stats.items():
        occurrences = result["occurrences"]

        candidate_coverage = (
            result["matched_total"] / result["annotated_total"]
            if result["annotated_total"]
            else 0.0
        )

        occurrence_coverage = (
            result["occurrences_with_match"] / occurrences
            if occurrences
            else 0.0
        )

        correct_coverage = (
            result["correct_matched"] / result["correct_total"]
            if result["correct_total"]
            else 0.0
        )

        correct_occurrence_coverage = (
            result["occurrences_with_correct"] / occurrences
            if occurrences
            else 0.0
        )

        average_candidates = (
            result["candidate_total"] / occurrences
            if occurrences
            else 0.0
        )

        print(name)
        print("-" * len(name))
        print(f"Occurrences evaluated: {occurrences}")
        print(f"Average candidates: {average_candidates:.2f}")
        print(
            "Candidate annotation coverage: "
            f"{candidate_coverage:.4f}"
        )
        print(
            "Occurrence match coverage: "
            f"{occurrence_coverage:.4f}"
        )
        print(
            "Correct-candidate coverage: "
            f"{correct_coverage:.4f}"
        )
        print(
            "Occurrence correct-candidate coverage: "
            f"{correct_occurrence_coverage:.4f}"
        )
        print()


def main():
    """Run candidate-generation ablation on validation data."""

    print("Loading DAMIR...")
    df = load_damir(DATA_PATH)

    print("Creating fixed train/validation/test split...")
    _, validation, _ = split_damir(df)

    print()
    print("Validation split")
    print("================")
    print(f"Occurrences: {validation['Id'].nunique()}")
    print(f"Candidate rows: {len(validation)}")

    print()
    print("Evaluating candidate-generation strategies...")

    stats = evaluate_strategy(validation)

    print_results(stats)

    print("The test split was not used in this experiment.")


if __name__ == "__main__":
    main()
