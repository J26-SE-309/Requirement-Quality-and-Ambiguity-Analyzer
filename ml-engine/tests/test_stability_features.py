"""Unit tests for Task 3 features; no models or training required."""

import pytest
from quality_ml.stability_features import FEATURE_NAMES, extract_stability_features

A = [1.0, 0.0, 0.0]
B = [0.0, 1.0, 0.0]


def test_identical_requirements():
    result = extract_stability_features(
        "The system shall save 5 records.", "The system shall save 5 records.", A, A)
    assert tuple(result) == FEATURE_NAMES
    assert result["semantic_cosine"] == pytest.approx(1.0)
    assert result["token_jaccard"] == pytest.approx(1.0)
    assert result["number_changed"] == 0.0


def test_changed_numeric_constraint():
    result = extract_stability_features(
        "The system shall save 5 records.", "The system shall save 10 records.", A, B)
    assert result["number_changed"] == 1.0
    assert result["semantic_cosine"] == pytest.approx(0.0)


def test_modal_negation_and_temporal_changes():
    result = extract_stability_features(
        "The system shall run before deployment.", "The system may not run after deployment.", A, A)
    assert result["modal_changed"] == 1.0
    assert result["negation_changed"] == 1.0
    assert result["temporal_changed"] == 1.0


def test_rejects_invalid_embedding():
    with pytest.raises(ValueError):
        extract_stability_features(
            "First requirement", "Second requirement", [0, 0], [0, 0])


def test_rejects_blank_text():
    with pytest.raises(ValueError):
        extract_stability_features("", "Second requirement", A, A)
