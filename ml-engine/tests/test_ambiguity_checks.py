import pytest
from quality_ml.ambiguity_checks import (
    find_candidate_antecedents,
    find_pronouns,
    get_ambiguity_signals,
)


def test_find_pronouns():
    text = "The vehicle shall send its status to the control unit."

    pronouns = find_pronouns(text)

    assert len(pronouns) == 1
    assert pronouns[0]["text"] == "its"
    assert pronouns[0]["lemma"] == "its"


def test_find_candidate_antecedents():
    text = "The controller shall communicate with the sensor before it starts."

    pronouns = find_pronouns(text)

    assert len(pronouns) == 1

    candidates = find_candidate_antecedents(
        text,
        pronouns[0]["position"],
    )

    candidate_texts = [candidate["text"] for candidate in candidates]

    assert "The controller" in candidate_texts
    assert "the sensor" in candidate_texts


def test_get_ambiguity_signals_detects_multiple_candidates():
    text = (
        "The controller shall communicate with the sensor "
        "before it starts the diagnostic process."
    )

    pronoun = find_pronouns(text)[0]

    result = get_ambiguity_signals(
        text,
        pronoun["position"],
    )

    assert result["pronoun"] == "it"
    assert len(result["candidates"]) == 2
    assert result["signals"]["lexical"] is True
    assert result["signals"]["anaphoric"] is True


def test_empty_text_is_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        find_pronouns("")


def test_invalid_pronoun_position_is_rejected():
    text = "The system shall process the request."

    with pytest.raises(ValueError, match="No token found"):
        get_ambiguity_signals(text, 999)
