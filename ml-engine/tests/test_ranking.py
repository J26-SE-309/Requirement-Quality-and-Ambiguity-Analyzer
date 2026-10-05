import pytest
from quality_ml.ranking import (
    calculate_combined_score,
    calculate_proximity_score,
    calculate_syntactic_score,
)


def test_proximity_score_prefers_closer_candidate():
    close = calculate_proximity_score(
        pronoun_position=10,
        candidate_position=8,
    )

    far = calculate_proximity_score(
        pronoun_position=10,
        candidate_position=1,
    )

    assert close > far
    assert 0.0 < close <= 1.0
    assert 0.0 < far <= 1.0


def test_proximity_score_is_high_for_adjacent_candidate():
    score = calculate_proximity_score(
        pronoun_position=10,
        candidate_position=9,
    )

    farther = calculate_proximity_score(
        pronoun_position=10,
        candidate_position=20,
    )

    assert score > farther
    assert score > 0.5

def test_syntactic_score_prefers_subject():
    subject = calculate_syntactic_score("nsubj")
    object_score = calculate_syntactic_score("obj")

    assert subject > object_score


def test_unknown_dependency_gets_default_score():
    score = calculate_syntactic_score("unknown")

    assert score == 0.5


def test_combined_score_uses_weights():
    score = calculate_combined_score(
        semantic_score=0.8,
        proximity_score=0.6,
        syntactic_score=1.0,
        semantic_weight=0.5,
        proximity_weight=0.3,
        syntactic_weight=0.2,
    )

    expected = (
        0.5 * 0.8
        + 0.3 * 0.6
        + 0.2 * 1.0
    )

    assert score == pytest.approx(expected)


def test_combined_score_rejects_zero_total_weight():
    with pytest.raises(
        ValueError,
        match="positive",
    ):
        calculate_combined_score(
            semantic_score=0.8,
            proximity_score=0.6,
            syntactic_score=1.0,
            semantic_weight=0.0,
            proximity_weight=0.0,
            syntactic_weight=0.0,
        )
