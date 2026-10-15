"""Candidate ranking features for ambiguity analysis."""

import math


def calculate_proximity_score(
    pronoun_position: int,
    candidate_position: int,
) -> float:
    """Convert token distance into a normalized proximity score.

    The parameter name ``pronoun_position`` is retained for backward
    compatibility with existing tests and callers.

    Although the name says "pronoun", the same function can also be
    used for demonstrative and relative references because the
    calculation only depends on token positions.
    """

    distance = abs(
        pronoun_position - candidate_position
    )

    return 1.0 / (
        1.0 + math.log1p(distance)
    )


def calculate_recency_score(
    reference_position: int,
    candidate_position: int,
) -> float:
    """Measure how recently a candidate occurs before a reference.

    Candidates closer to the reference receive a higher score.
    Candidates occurring at or after the reference receive zero.
    """

    if reference_position <= 0:
        return 0.0

    if candidate_position < 0:
        return 0.0

    if candidate_position >= reference_position:
        return 0.0

    return candidate_position / reference_position


def calculate_syntactic_score(
    dependency: str,
) -> float:
    """Assign a normalized score based on candidate dependency role."""

    scores = {
        "nsubj": 1.0,
        "nsubjpass": 1.0,
        "csubj": 0.95,
        "dobj": 0.85,
        "obj": 0.85,
        "iobj": 0.80,
        "pobj": 0.75,
        "attr": 0.75,
        "oprd": 0.70,
        "conj": 0.65,
        "appos": 0.65,
    }

    return scores.get(
        dependency,
        0.5,
    )


def calculate_combined_score(
    semantic_score: float,
    proximity_score: float,
    syntactic_score: float,
    semantic_weight: float = 0.70,
    proximity_weight: float = 0.15,
    syntactic_weight: float = 0.15,
) -> float:
    """Combine semantic, proximity and syntactic evidence.

    This produces the base ranking score.

    Linguistic compatibility and recency are applied afterwards by
    ``calculate_linguistic_ranking_score``.
    """

    total_weight = (
        semantic_weight
        + proximity_weight
        + syntactic_weight
    )

    if total_weight <= 0:
        raise ValueError(
            "Ranking weights must have a positive total."
        )

    score = (
        semantic_weight * semantic_score
        + proximity_weight * proximity_score
        + syntactic_weight * syntactic_score
    ) / total_weight

    return max(
        0.0,
        min(1.0, score),
    )


def calculate_linguistic_ranking_score(
    base_score: float,
    compatibility_score: float,
    recency_score: float,
    compatibility_weight: float = 0.25,
    recency_weight: float = 0.05,
) -> float:
    """Combine base ranking with compatibility and recency.

    Strongly incompatible candidates receive a score of zero.

    Compatible candidates are ranked using a normalized weighted
    combination. This avoids additive score saturation where several
    strong candidates could all become exactly 1.0 and therefore lose
    meaningful ranking differences.
    """

    base_score = max(
        0.0,
        min(1.0, base_score),
    )

    compatibility_score = max(
        0.0,
        min(1.0, compatibility_score),
    )

    recency_score = max(
        0.0,
        min(1.0, recency_score),
    )

    if compatibility_score <= 0.10:
        return 0.0

    base_weight = max(
        0.0,
        1.0
        - compatibility_weight
        - recency_weight,
    )

    total_weight = (
        base_weight
        + compatibility_weight
        + recency_weight
    )

    if total_weight <= 0.0:
        return base_score

    score = (
        base_weight * base_score
        + compatibility_weight
        * compatibility_score
        + recency_weight
        * recency_score
    ) / total_weight

    return max(
        0.0,
        min(1.0, score),
    )


def calculate_linguistic_adjustment(
    base_score: float,
    compatibility_score: float,
    weight: float = 0.05,
) -> float:
    """Apply the original compatibility adjustment.

    Retained for backward compatibility with existing tests and
    callers.

    New Task 2 ranking should use
    ``calculate_linguistic_ranking_score``.
    """

    if not 0.0 <= base_score <= 1.0:
        raise ValueError(
            "Base score must be between 0 and 1."
        )

    if not 0.0 <= compatibility_score <= 1.0:
        raise ValueError(
            "Compatibility score must be between 0 and 1."
        )

    if not 0.0 <= weight <= 1.0:
        raise ValueError(
            "Adjustment weight must be between 0 and 1."
        )

    adjusted_score = (
        base_score
        + weight
        * (
            compatibility_score - 0.5
        )
    )

    return max(
        0.0,
        min(1.0, adjusted_score),
    )
