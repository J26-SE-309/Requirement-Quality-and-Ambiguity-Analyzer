"""Candidate ranking features for ambiguity analysis."""

import math


def calculate_proximity_score(
    pronoun_position: int,
    candidate_position: int,
) -> float:
    """Convert token distance into a normalized proximity score."""

    distance = abs(
        pronoun_position - candidate_position
    )

    return 1.0 / (1.0 + math.log1p(distance))


def calculate_syntactic_score(
    dependency: str,
) -> float:
    """Assign a simple score based on candidate dependency."""

    scores = {
        "nsubj": 1.0,
        "nsubjpass": 1.0,
        "dobj": 0.8,
        "obj": 0.8,
        "pobj": 0.7,
        "attr": 0.7,
        "conj": 0.6,
    }

    return scores.get(
        dependency,
        0.5,
    )


def calculate_combined_score(
    semantic_score: float,
    proximity_score: float,
    syntactic_score: float,
    semantic_weight: float = 0.5,
    proximity_weight: float = 0.3,
    syntactic_weight: float = 0.2,
) -> float:
    """Combine semantic, proximity and syntactic scores."""

    total_weight = (
        semantic_weight
        + proximity_weight
        + syntactic_weight
    )

    if total_weight <= 0:
        raise ValueError(
            "Ranking weights must have a positive total."
        )

    return (
        semantic_weight * semantic_score
        + proximity_weight * proximity_score
        + syntactic_weight * syntactic_score
    ) / total_weight
