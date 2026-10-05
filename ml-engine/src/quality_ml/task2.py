"""Context-aware pronoun resolution using the fine-tuned SBERT model."""

from quality_ml.ambiguity_checks import (
    find_pronouns,
    get_ambiguity_signals,
)
from quality_ml.interpretations import generate_interpretations
from quality_ml.ranking import calculate_combined_score, calculate_proximity_score
from quality_ml.semantic_similarity import rank_candidates, similarity_gap


class Task2AmbiguityAnalyzer:
    """Rank pronoun antecedents using semantic similarity and proximity."""

    def analyze(self, text: str) -> dict:
        """Analyze pronoun antecedents and return ranked interpretations."""

        if not text or not text.strip():
            raise ValueError("Requirement text must not be empty.")

        pronouns = find_pronouns(text)
        analyses = []

        for pronoun in pronouns:
            signals = get_ambiguity_signals(
                text,
                pronoun["position"],
            )

            interpretations = generate_interpretations(
                text,
                pronoun,
                signals["candidates"],
            )

            ranked_candidates = rank_candidates(
                text,
                interpretations,
            )

            for candidate in ranked_candidates:
                candidate_position = next(
                    (
                        item["root_position"]
                        for item in signals["candidates"]
                        if item["text"] == candidate["candidate"]
                        and item["start"] == candidate["candidate_start"]
                    ),
                    pronoun["position"],
                )

                proximity_score = calculate_proximity_score(
                    pronoun["position"],
                    candidate_position,
                )

                candidate["proximity"] = proximity_score
                candidate["combined_score"] = calculate_combined_score(
                    semantic_score=candidate["similarity"],
                    proximity_score=proximity_score,
                    syntactic_score=0.0,
                )

            ranked_candidates.sort(
                key=lambda candidate: candidate["combined_score"],
                reverse=True,
            )

            for rank, candidate in enumerate(ranked_candidates, start=1):
                candidate["rank"] = rank

            gap = similarity_gap(ranked_candidates)
            candidate_count = len(ranked_candidates)

            if ranked_candidates:
                best_candidate = ranked_candidates[0]
                confidence = best_candidate["combined_score"]
            else:
                best_candidate = None
                confidence = 0.0

            if len(ranked_candidates) >= 2:
                second_score = ranked_candidates[1]["combined_score"]
                competition_score = max(
                    0.0,
                    1.0 - (best_candidate["combined_score"] - second_score),
                )
            else:
                competition_score = 0.0

            is_ambiguous = (
                candidate_count >= 2
                and competition_score >= 0.95
            )

            analyses.append(
                {
                    "pronoun": pronoun["text"],
                    "position": pronoun["position"],
                    "span": {
                        "start": pronoun["start"],
                        "end": pronoun["end"],
                    },
                    "candidate_count": candidate_count,
                    "is_ambiguous": is_ambiguous,
                    "signals": {
                        "lexical": signals["signals"]["lexical"],
                        "syntactic": signals["signals"]["syntactic"],
                        "semantic": candidate_count >= 2,
                        "anaphoric": signals["signals"]["anaphoric"],
                    },
                    "candidate_interpretations": ranked_candidates,
                    "similarity_gap": gap,
                    "best_candidate": (
                        best_candidate["candidate"]
                        if best_candidate
                        else None
                    ),
                    "confidence": confidence,
                    "competition_score": competition_score,
                }
            )

        ambiguous_count = sum(
            analysis["is_ambiguous"]
            for analysis in analyses
        )

        ambiguity_score = (
            ambiguous_count / len(analyses)
            if analyses
            else 0.0
        )

        return {
            "is_ambiguous": ambiguous_count > 0,
            "score": ambiguity_score,
            "pronouns": analyses,
        }
