from quality_ml.ambiguity_checks import (
    find_candidate_antecedents,
    find_pronouns,
    find_reference_expressions,
    get_ambiguity_signals,
)


def test_find_pronouns():
    text = (
        "The system shall update the server "
        "before it starts the backup process."
    )

    pronouns = find_pronouns(text)

    assert pronouns
    assert any(
        pronoun["text"] == "it"
        for pronoun in pronouns
    )


def test_find_reference_expressions():
    text = (
        "The system shall update the server "
        "before it starts the backup process."
    )

    references = find_reference_expressions(text)

    assert references

    reference = next(
        reference
        for reference in references
        if reference["text"] == "it"
    )

    assert reference["kind"] == "pronoun"
    assert reference["part_of_speech"] == "PRON"
    assert reference["dependency"]
    assert reference["head"]


def test_find_demonstrative_reference():
    text = (
        "The system stores the configuration. "
        "This shall be backed up daily."
    )

    references = find_reference_expressions(text)

    demonstratives = [
        reference
        for reference in references
        if reference["kind"] == "demonstrative"
    ]

    assert demonstratives
    assert demonstratives[0]["text"] == "This"


def test_find_relative_reference():
    text = (
        "The service shall use the database "
        "which stores the configuration."
    )

    references = find_reference_expressions(text)

    relatives = [
        reference
        for reference in references
        if reference["kind"] == "relative"
    ]

    assert relatives
    assert relatives[0]["text"] == "which"


def test_find_candidate_antecedents():
    text = (
        "The system shall update the server "
        "and the database before it starts."
    )

    pronouns = find_pronouns(text)

    reference = next(
        pronoun
        for pronoun in pronouns
        if pronoun["text"] == "it"
    )

    candidates = find_candidate_antecedents(
        text,
        reference["position"],
    )

    assert candidates
    assert all(
        candidate["root_position"] < reference["position"]
        for candidate in candidates
    )


def test_ambiguity_signals_for_pronoun():
    text = (
        "The system shall update the server "
        "and the database before it starts."
    )

    pronouns = find_pronouns(text)

    reference = next(
        pronoun
        for pronoun in pronouns
        if pronoun["text"] == "it"
    )

    signals = get_ambiguity_signals(
        text,
        reference["position"],
    )

    assert signals["reference"] == "it"
    assert signals["reference_kind"] == "pronoun"
    assert signals["pronoun"] == "it"
    assert signals["candidates"]

    assert "lexical" in signals["signals"]
    assert "syntactic" in signals["signals"]
    assert "anaphoric" in signals["signals"]


def test_ambiguity_signals_for_demonstrative():
    text = (
        "The system stores the configuration. "
        "This shall be backed up daily."
    )

    references = find_reference_expressions(text)

    reference = next(
        reference
        for reference in references
        if reference["kind"] == "demonstrative"
    )

    signals = get_ambiguity_signals(
        text,
        reference["position"],
    )

    assert signals["reference"] == "This"
    assert signals["reference_kind"] == "demonstrative"
    assert signals["pronoun"] is None
    assert "candidates" in signals


def test_ambiguity_signals_for_relative_reference():
    text = (
        "The service shall use the database "
        "which stores the configuration."
    )

    references = find_reference_expressions(text)

    reference = next(
        reference
        for reference in references
        if reference["kind"] == "relative"
    )

    signals = get_ambiguity_signals(
        text,
        reference["position"],
    )

    assert signals["reference"] == "which"
    assert signals["reference_kind"] == "relative"
    assert signals["pronoun"] is None
    assert "candidates" in signals


def test_empty_requirement_is_rejected():
    try:
        find_reference_expressions("")
    except ValueError as exc:
        assert str(exc) == "Requirement text must not be empty."
    else:
        raise AssertionError(
            "Expected ValueError for empty requirement."
        )
