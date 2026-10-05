"""Context-aware ambiguity and vagueness analysis."""

from quality_ml.ambiguity_checks import (
    find_pronouns,
    get_ambiguity_signals,
)
from quality_ml.interpretations import generate_interpretations
from quality_ml.semantic_similarity import (
    rank_candidates,
    similarity_gap,
)

SEMANTIC_GAP_THRESHOLD = 0.05


class Task2AmbiguityAnalyzer:
    """Combine spaCy ambiguity signals with SBERT semantic comparison."""

    def analyze(self, text: str) -> dict:
        """Analyze pronoun ambiguity in a requirement."""

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

            gap = similarity_gap(ranked_candidates)

            candidate_count = len(ranked_candidates)

            lexical_ambiguity = candidate_count >= 2

            syntactic_ambiguity = signals["signals"]["syntactic"]

            semantic_ambiguity = (
                candidate_count >= 2
                and gap < SEMANTIC_GAP_THRESHOLD
            )

            anaphoric_ambiguity = signals["signals"]["anaphoric"]

            ambiguity_evidence = sum(
                [
                    lexical_ambiguity,
                    syntactic_ambiguity,
                    semantic_ambiguity,
                    anaphoric_ambiguity,
                ]
            )

            is_ambiguous = (
                candidate_count >= 2
                and ambiguity_evidence >= 2
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
                        "lexical": lexical_ambiguity,
                        "syntactic": syntactic_ambiguity,
                        "semantic": semantic_ambiguity,
                        "anaphoric": anaphoric_ambiguity,
                    },
                    "candidate_interpretations": ranked_candidates,
                    "similarity_gap": gap,
                }
            )

        ambiguous_count = sum(
            analysis["is_ambiguous"]
            for analysis in analyses
        )

        if analyses:
            ambiguity_score = ambiguous_count / len(analyses)
        else:
            ambiguity_score = 0.0

        return {
            "is_ambiguous": ambiguous_count > 0,
            "score": ambiguity_score,
            "pronouns": analyses,
        }
