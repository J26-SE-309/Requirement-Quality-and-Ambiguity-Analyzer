import pytest
from quality_ml.task2 import Task2AmbiguityAnalyzer


def test_ambiguous_pronoun_has_multiple_interpretations():
    text = (
        "The controller shall communicate with the sensor "
        "before it starts the diagnostic process."
    )

    result = Task2AmbiguityAnalyzer().analyze(text)

    assert result["is_ambiguous"] is True
    assert result["score"] > 0

    assert len(result["pronouns"]) == 1

    pronoun = result["pronouns"][0]

    assert pronoun["pronoun"] == "it"
    assert pronoun["candidate_count"] >= 2
    assert pronoun["is_ambiguous"] is True

    assert pronoun["signals"]["lexical"] is True
    assert pronoun["signals"]["semantic"] is True
    assert pronoun["signals"]["anaphoric"] is True

    assert len(pronoun["candidate_interpretations"]) >= 2

    similarities = [
        candidate["similarity"]
        for candidate in pronoun["candidate_interpretations"]
    ]

    assert similarities == sorted(
        similarities,
        reverse=True,
    )


def test_requirement_without_pronouns_is_not_ambiguous():
    text = "The system shall process the request within two seconds."

    result = Task2AmbiguityAnalyzer().analyze(text)

    assert result["is_ambiguous"] is False
    assert result["score"] == 0.0
    assert result["pronouns"] == []


def test_multiple_pronouns_are_analyzed():
    text = (
        "The controller shall send its status to the server "
        "when it completes the operation."
    )

    result = Task2AmbiguityAnalyzer().analyze(text)

    assert len(result["pronouns"]) == 2
    assert result["score"] >= 0.0

    pronoun_texts = [
        analysis["pronoun"]
        for analysis in result["pronouns"]
    ]

    assert pronoun_texts == ["its", "it"]


def test_empty_requirement_is_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        Task2AmbiguityAnalyzer().analyze("")
