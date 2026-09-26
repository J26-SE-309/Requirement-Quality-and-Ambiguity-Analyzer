"""Version 1 of the analyzer API.

Until the DeBERTa, SBERT and stability models are plugged in, the endpoints return
placeholder results (model_version "stub") so the gateway and frontend can be built
against the real contract.
"""

from datetime import UTC, datetime

from fastapi import APIRouter

from app.schemas import (
    AmbiguityResult,
    QualityResult,
    RequirementAnalysis,
    RequirementIn,
    StabilityResult,
    VaguenessResult,
)

router = APIRouter(tags=["requirement quality"])


def _analyse(requirement: RequirementIn) -> RequirementAnalysis:
    return RequirementAnalysis(
        requirement_id=requirement.requirement_id,
        quality=QualityResult(label="not_analysed", score=0.5),
        ambiguity=AmbiguityResult(is_ambiguous=False, score=0.0),
        vagueness=VaguenessResult(count=0),
        stability=StabilityResult(label="stable", score=1.0),
        confidence=0.0,
        explanation="Placeholder result: the analysis models are not implemented yet.",
        model_version="stub",
        analysed_at=datetime.now(UTC),
    )


@router.post("/analyze", response_model=RequirementAnalysis)
def analyze(requirement: RequirementIn) -> RequirementAnalysis:
    """Analyse one raw requirement for quality, ambiguity, vagueness and interpretation stability."""
    return _analyse(requirement)


@router.post("/analyze/batch", response_model=list[RequirementAnalysis])
def analyze_batch(requirements: list[RequirementIn]) -> list[RequirementAnalysis]:
    """Analyse several requirements in one call, in the order given (NFR-06)."""
    return [_analyse(requirement) for requirement in requirements]
