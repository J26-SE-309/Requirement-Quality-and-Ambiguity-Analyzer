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
    VaguenessResult,
)

router = APIRouter(tags=["requirement quality"])

_task1_analyzer = None


def _get_task1_analyzer():
    """Load the Task 1 analyzer only when an analysis is requested."""
    global _task1_analyzer

    if _task1_analyzer is None:
        from quality_ml.task1 import Task1QualityAnalyzer

        _task1_analyzer = Task1QualityAnalyzer()

    return _task1_analyzer


def _analyse(requirement: RequirementIn) -> RequirementAnalysis:
    """Run Task 1 quality analysis and keep Task 2/3 as placeholders."""
    task1_result = _get_task1_analyzer().analyze(requirement.text)

    issues = [
        QualityIssue(
            type=issue["type"],
            message=issue["message"],
        )
        for issue in task1_result["issues"]
    ]

    missing_elements = task1_result["structural_completeness"][
        "missing_elements"
    ]

    if missing_elements:
        explanation = (
            "Task 1 identified structural completeness issues: "
            + ", ".join(missing_elements)
            + "."
        )
    elif task1_result["model_label"] == "defect":
        explanation = (
            "Task 1 identified the requirement as potentially defective "
            "using the fine-tuned QuRE DeBERTa model."
        )
    else:
        explanation = (
            "Task 1 found the requirement structurally complete and "
            "classified it as acceptable by the fine-tuned QuRE DeBERTa model."
        )

    return RequirementAnalysis(
        requirement_id=requirement.requirement_id,
        quality=QualityResult(
            label=task1_result["label"],
            score=task1_result["score"],
            issues=issues,
        ),
        ambiguity=AmbiguityResult(
            is_ambiguous=False,
            score=0.0,
            spans=[],
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
        model_version="task1-exp06-deberta-spacy",
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
