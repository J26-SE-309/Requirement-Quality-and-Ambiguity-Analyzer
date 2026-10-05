"""Version 1 of the analyzer API."""

from datetime import UTC, datetime

from fastapi import APIRouter

from app.schemas import (
    AmbiguityResult,
    QualityIssue,
    QualityResult,
    RequirementAnalysis,
    RequirementIn,
    StabilityResult,
    TextSpan,
    VaguenessResult,
)

router = APIRouter(tags=["requirement quality"])

_task1_analyzer = None
_task2_analyzer = None


def _get_task1_analyzer():
    """Load the Task 1 analyzer only when an analysis is requested."""
    global _task1_analyzer

    if _task1_analyzer is None:
        from quality_ml.task1 import Task1QualityAnalyzer

        _task1_analyzer = Task1QualityAnalyzer()

    return _task1_analyzer


def _get_task2_analyzer():
    """Load the Task 2 analyzer only when an analysis is requested."""
    global _task2_analyzer

    if _task2_analyzer is None:
        from quality_ml.task2 import Task2AmbiguityAnalyzer

        _task2_analyzer = Task2AmbiguityAnalyzer()

    return _task2_analyzer


def _analyse(requirement: RequirementIn) -> RequirementAnalysis:
    """Run Task 1 and Task 2 analysis while keeping Task 3 as a placeholder."""
    task1_result = _get_task1_analyzer().analyze(requirement.text)
    task2_result = _get_task2_analyzer().analyze(requirement.text)

    issues = [
        QualityIssue(
            type=issue["type"],
            message=issue["message"],
            terms=issue.get("terms", []),
        )
        for issue in task1_result["issues"]
    ]

    missing_elements = task1_result["structural_completeness"][
        "missing_elements"
    ]

    vague_terms = task1_result.get("vague_terms", [])

    if missing_elements:
        explanation = (
            "The requirement needs additional information. "
            "The following elements are missing: "
            + ", ".join(missing_elements)
            + "."
        )
    elif vague_terms:
        explanation = (
            "The requirement contains potentially vague wording: "
            + ", ".join(f'"{term}"' for term in vague_terms)
            + ". These terms should be replaced with specific, "
            "measurable criteria where possible."
        )
    elif task1_result["model_label"] == "defect":
        explanation = (
            "The requirement may contain quality issues. "
            "Consider reviewing it for clearer and more measurable wording."
        )
    else:
        explanation = (
            "The requirement is structurally complete and was assessed as acceptable."
        )

    ambiguity_spans = [
        TextSpan(
            start=analysis["span"]["start"],
            end=analysis["span"]["end"],
            text=analysis["pronoun"],
        )
        for analysis in task2_result["pronouns"]
        if analysis["is_ambiguous"]
    ]

    return RequirementAnalysis(
        requirement_id=requirement.requirement_id,
        quality=QualityResult(
            label=task1_result["label"],
            score=task1_result["score"],
            issues=issues,
        ),
        ambiguity=AmbiguityResult(
            is_ambiguous=task2_result["is_ambiguous"],
            score=task2_result["score"],
            spans=ambiguity_spans,
        ),
        vagueness=VaguenessResult(
            count=0,
            terms=[],
        ),
        stability=StabilityResult(
            label="stable",
            score=1.0,
        ),
        confidence=task1_result["score"],
        explanation=explanation,
        model_version="task1-exp06-deberta-spacy+task2-exp04-minilm",
        analysed_at=datetime.now(UTC),
    )


@router.post("/analyze", response_model=RequirementAnalysis)
def analyze(requirement: RequirementIn) -> RequirementAnalysis:
    """Analyse one raw requirement."""
    return _analyse(requirement)


@router.post("/analyze/batch", response_model=list[RequirementAnalysis])
def analyze_batch(
    requirements: list[RequirementIn],
) -> list[RequirementAnalysis]:
    """Analyse several requirements in one call."""
    return [_analyse(requirement) for requirement in requirements]
