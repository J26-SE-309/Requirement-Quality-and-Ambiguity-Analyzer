"""Compare current and no-conj candidate generation on DAMIR validation."""

from pathlib import Path

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
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

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
    """Normalize text for annotation matching."""

    return " ".join(str(text).strip().lower().split())


def prepare_candidates(
    context: str,
    pronoun: dict,
    no_conj: bool,
):
    """Generate candidates using one of the two strategies."""

    candidates = find_candidate_antecedents(
        context,
        pronoun["position"],
    )

    if no_conj:
        candidates = [
            candidate
            for candidate in candidates
            if candidate["dependency"] != "conj"
        ]

    return candidates


def evaluate_strategy(validation, no_conj: bool):
    """Evaluate ranking and ambiguity metrics for one strategy."""

    ranking_evaluated = 0
    top1_correct = 0
    reciprocal_rank_sum = 0.0

    ambiguity_true = []
    ambiguity_predicted = []

    candidate_total = 0
    matched_total = 0
    correct_total = 0
    correct_matched_total = 0

    groups = validation.groupby(
        ["Id", "Context", "Pronoun", "Position", "AckUnack"],
        sort=False,
    )

    for (
        occurrence_id,
        context,
        pronoun_text,
        position,
        ack_unack,
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

        candidates = prepare_candidates(
            context,
            pronoun,
            no_conj,
        )

        candidate_total += len(candidates)

        annotated = {
            normalize(row["Candidate Antecedent"])
            for _, row in group.iterrows()
        }

        correct = {
            normalize(row["Candidate Antecedent"])
            for _, row in group.iterrows()
            if normalize(row["ResolvedAs"]) == "correct"
        }

        extracted_texts = {
            normalize(candidate["text"])
            for candidate in candidates
        }

        matched = annotated & extracted_texts
        correct_matched = correct & extracted_texts

        matched_total += len(matched)
        correct_total += len(correct)
        correct_matched_total += len(correct_matched)

        if not candidates:
            predicted_ambiguous = False
        else:
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
                        **ranked_candidate,
                        "combined_score": combined,
                    }
                )

            scored.sort(
                key=lambda candidate: candidate["combined_score"],
                reverse=True,
            )

            ranking_evaluated += 1

            correct_rank = None

            for rank, candidate in enumerate(scored, start=1):
                if normalize(candidate["candidate"]) in correct:
                    correct_rank = rank
                    break

            if correct_rank == 1:
                top1_correct += 1

            if correct_rank is not None:
                reciprocal_rank_sum += 1.0 / correct_rank

            if len(scored) < 2:
                predicted_ambiguous = False
            else:
                top_score = scored[0]["combined_score"]
                second_score = scored[1]["combined_score"]

                predicted_ambiguous = (
                    top_score - second_score < 0.05
                )

        actual_ambiguous = ack_unack in {
            "Ack",
            "Unack",
        }

        ambiguity_true.append(int(actual_ambiguous))
        ambiguity_predicted.append(int(predicted_ambiguous))

    occurrences = len(ambiguity_true)

    candidate_coverage = (
        matched_total / sum(
            len(
                {
                    normalize(row["Candidate Antecedent"])
                    for _, row in group.iterrows()
                }
            )
            for _, group in groups
        )
        if occurrences
        else 0.0
    )

    correct_coverage = (
        correct_matched_total / correct_total
        if correct_total
        else 0.0
    )

    top1 = (
        top1_correct / ranking_evaluated
        if ranking_evaluated
        else 0.0
    )

    mrr = (
        reciprocal_rank_sum / ranking_evaluated
        if ranking_evaluated
        else 0.0
    )

    return {
        "occurrences": occurrences,
        "average_candidates": candidate_total / occurrences,
        "candidate_coverage": candidate_coverage,
        "correct_coverage": correct_coverage,
        "ranking_evaluated": ranking_evaluated,
        "top1": top1,
        "mrr": mrr,
        "accuracy": accuracy_score(
            ambiguity_true,
            ambiguity_predicted,
        ),
        "precision": precision_score(
            ambiguity_true,
            ambiguity_predicted,
            zero_division=0,
        ),
        "recall": recall_score(
            ambiguity_true,
            ambiguity_predicted,
            zero_division=0,
        ),
        "f1": f1_score(
            ambiguity_true,
            ambiguity_predicted,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            ambiguity_true,
            ambiguity_predicted,
        ),
    }


def print_result(name: str, result: dict):
    """Print evaluation results."""

    print()
    print(name)
    print("=" * len(name))

    print(f"Occurrences: {result['occurrences']}")
    print(
        f"Average candidates: "
        f"{result['average_candidates']:.2f}"
    )
    print(
        f"Candidate coverage: "
        f"{result['candidate_coverage']:.4f}"
    )
    print(
        f"Correct-candidate coverage: "
        f"{result['correct_coverage']:.4f}"
    )

    print()
    print("Candidate ranking")
    print("-----------------")
    print(
        f"Evaluated: "
        f"{result['ranking_evaluated']}"
    )
    print(
        f"Top-1: "
        f"{result['top1']:.4f}"
    )
    print(
        f"MRR: "
        f"{result['mrr']:.4f}"
    )

    print()
    print("Ambiguity detection")
    print("--------------------")
    print(
        f"Accuracy: "
        f"{result['accuracy']:.4f}"
    )
    print(
        f"Precision: "
        f"{result['precision']:.4f}"
    )
    print(
        f"Recall: "
        f"{result['recall']:.4f}"
    )
    print(
        f"F1: "
        f"{result['f1']:.4f}"
    )

    print()
    print("Confusion matrix")
    print("----------------")
    print(result["confusion_matrix"])


def main():
    """Compare the two candidate-generation strategies."""

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
    print(
        "Using fixed ranking weights: "
        "semantic=0.7, proximity=0.2, syntactic=0.1"
    )

    print()
    print("Evaluating current strategy...")
    current = evaluate_strategy(
        validation,
        no_conj=False,
    )

    print("Evaluating no-conj strategy...")
    no_conj = evaluate_strategy(
        validation,
        no_conj=True,
    )

    print_result(
        "A_current",
        current,
    )

    print_result(
        "C_no_conj",
        no_conj,
    )

    print()
    print("Comparison")
    print("==========")
    print(
        f"Average candidates: "
        f"{current['average_candidates']:.2f} -> "
        f"{no_conj['average_candidates']:.2f}"
    )
    print(
        f"Correct-candidate coverage: "
        f"{current['correct_coverage']:.4f} -> "
        f"{no_conj['correct_coverage']:.4f}"
    )
    print(
        f"Ranking Top-1: "
        f"{current['top1']:.4f} -> "
        f"{no_conj['top1']:.4f}"
    )
    print(
        f"Ranking MRR: "
        f"{current['mrr']:.4f} -> "
        f"{no_conj['mrr']:.4f}"
    )
    print(
        f"Ambiguity F1: "
        f"{current['f1']:.4f} -> "
        f"{no_conj['f1']:.4f}"
    )

    print()
    print("The test split was not used.")


if __name__ == "__main__":
    main()
