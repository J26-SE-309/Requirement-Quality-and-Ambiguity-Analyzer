import pytest
from quality_ml.task2 import Task2AmbiguityAnalyzer


def test_ambiguous_pronoun_has_multiple_interpretations():
    analyzer = Task2AmbiguityAnalyzer()

    result = analyzer.analyze(
        "The system shall update the server and the database "
        "before it starts the backup process."
    )

    assert result["references"]

    pronoun_analysis = next(
        analysis
        for analysis in result["references"]
        if analysis["reference_kind"] == "pronoun"
    )

    assert pronoun_analysis["reference"] == "it"
    assert pronoun_analysis["candidate_count"] >= 2
    assert pronoun_analysis["candidate_interpretations"]

    ranking_scores = [
        candidate["ranking_score"]
        for candidate in pronoun_analysis["candidate_interpretations"]
    ]

    assert ranking_scores == sorted(
        ranking_scores,
        reverse=True,
    )

    for candidate in pronoun_analysis["candidate_interpretations"]:
        assert 0.0 <= candidate["combined_score"] <= 1.0
        assert 0.0 <= candidate["ranking_score"] <= 1.0

    ranks = [
        candidate["rank"]
        for candidate in pronoun_analysis["candidate_interpretations"]
    ]

    assert ranks == list(range(1, len(ranks) + 1))

    assert (
        pronoun_analysis["best_candidate"]
        == pronoun_analysis["candidate_interpretations"][0]["candidate"]
    )


def test_requirement_without_pronouns_is_not_ambiguous():
    analyzer = Task2AmbiguityAnalyzer()

    result = analyzer.analyze(
        "The system shall generate a report."
    )

    assert result["references"] == []
    assert result["pronouns"] == []
    assert result["is_ambiguous"] is False
    assert result["score"] == 0.0


def test_multiple_pronouns_are_analyzed():
    analyzer = Task2AmbiguityAnalyzer()

    result = analyzer.analyze(
        "The user sends the request to the service and "
        "it validates it before processing."
    )

    pronouns = result["pronouns"]

    assert len(pronouns) >= 2

    for analysis in pronouns:
        assert analysis["reference_kind"] == "pronoun"
        assert "candidate_interpretations" in analysis
        assert "best_candidate" in analysis
        assert "confidence" in analysis
        assert "competition_score" in analysis


def test_empty_requirement_is_rejected():
    analyzer = Task2AmbiguityAnalyzer()

    with pytest.raises(ValueError):
        analyzer.analyze("")


def test_demonstrative_reference_is_analyzed():
    analyzer = Task2AmbiguityAnalyzer()

    result = analyzer.analyze(
        "The system stores the configuration in the database. "
        "This shall be backed up daily."
    )

    references = result["references"]

    demonstratives = [
        analysis
        for analysis in references
        if analysis["reference_kind"] == "demonstrative"
    ]

    assert demonstratives

    analysis = demonstratives[0]

    assert analysis["reference"] == "This"
    assert analysis["candidate_interpretations"]
    assert analysis["candidate_count"] >= 1


def test_relative_reference_is_analyzed():
    analyzer = Task2AmbiguityAnalyzer()

    result = analyzer.analyze(
        "The service shall use the database "
        "which stores the configuration."
    )

    references = result["references"]

    relatives = [
        analysis
        for analysis in references
        if analysis["reference_kind"] == "relative"
    ]

    assert relatives

    analysis = relatives[0]

    assert analysis["reference"] == "which"
    assert analysis["candidate_interpretations"]
    assert analysis["candidate_count"] >= 1


def test_reference_contains_linguistic_information():
    analyzer = Task2AmbiguityAnalyzer()

    result = analyzer.analyze(
        "The administrator shall review the report "
        "before they approve the deployment."
    )

    reference = next(
        analysis
        for analysis in result["references"]
        if analysis["reference"] == "they"
    )

    assert reference["linguistic"]["part_of_speech"]
    assert reference["linguistic"]["dependency"]
    assert reference["linguistic"]["head"]
    assert isinstance(
        reference["linguistic"]["head_position"],
        int,
    )


def test_reference_competition_signal_is_reported():
    analyzer = Task2AmbiguityAnalyzer()

    result = analyzer.analyze(
        "The system shall update the server and the database "
        "before it starts the backup process."
    )

    reference = next(
        analysis
        for analysis in result["references"]
        if analysis["reference"] == "it"
    )

    assert 0.0 <= reference["confidence"] <= 1.0
    assert 0.0 <= reference["competition_score"] <= 1.0

    if reference["candidate_count"] < 2:
        assert reference["competition_score"] == 0.0


def test_coordinated_candidate_is_ranked_for_plural_reference():
    analyzer = Task2AmbiguityAnalyzer()

    result = analyzer.analyze(
        "The manager reviewed the developer and the tester "
        "before they submitted the report."
    )

    reference = next(
        analysis
        for analysis in result["references"]
        if analysis["reference"] == "they"
    )

    assert (
        reference["best_candidate"]
        == "the developer and the tester"
    )


def test_relative_reference_prefers_payment_service():
    analyzer = Task2AmbiguityAnalyzer()

    result = analyzer.analyze(
        "The application sends the transaction "
        "to the payment service which validates the request."
    )

    reference = next(
        analysis
        for analysis in result["references"]
        if analysis["reference"] == "which"
    )

    assert reference["best_candidate"] == "the payment service"


def test_relative_person_reference_prefers_engineer():
    analyzer = Task2AmbiguityAnalyzer()

    result = analyzer.analyze(
        "The project manager discussed the requirements "
        "with the software engineer who updated the document "
        "before she submitted it."
    )

    reference = next(
        analysis
        for analysis in result["references"]
        if analysis["reference"] == "who"
    )

    assert reference["best_candidate"] == "the software engineer"


def test_person_reference_uses_linguistic_compatibility():
    analyzer = Task2AmbiguityAnalyzer()

    result = analyzer.analyze(
        "The project manager discussed the requirements "
        "with the software engineer who updated the document "
        "before she submitted it."
    )

    reference = next(
        analysis
        for analysis in result["references"]
        if analysis["reference"] == "she"
    )

    assert reference["best_candidate"] == "the software engineer"


def test_object_reference_prefers_document():
    analyzer = Task2AmbiguityAnalyzer()

    result = analyzer.analyze(
        "The project manager discussed the requirements "
        "with the software engineer who updated the document "
        "before she submitted it."
    )

    reference = next(
        analysis
        for analysis in result["references"]
        if analysis["reference"] == "it"
    )

    assert reference["best_candidate"] == "the document"
