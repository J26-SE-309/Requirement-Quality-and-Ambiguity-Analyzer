
"""Request and response models: The API contract of this service.

They mirror the JSON Schemas in Synapse-Web/contracts/requirement-quality;
change both together.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class RequirementIn(BaseModel):
    requirement_id: str
    project_id: str | None = None
    text: str = Field(min_length=1)


class TextSpan(BaseModel):
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    text: str


class QualityIssue(BaseModel):
    type: str = Field(
        description=(
            "Defect category, e.g. incomplete, inconsistent, "
            "grammar, missing_information"
        )
    )
    message: str
    span: TextSpan | None = None
    terms: list[str] = []


class QualityResult(BaseModel):
    label: str = Field(description="Overall quality category")
    score: float = Field(ge=0, le=1, description="1 = high quality")
    issues: list[QualityIssue] = []


class AmbiguityCandidate(BaseModel):
    candidate: str
    similarity: float = Field(ge=0, le=1)
    proximity: float = Field(ge=0, le=1)
    combined_score: float = Field(ge=0, le=1)
    rank: int = Field(ge=1)
    candidate_start: int = Field(ge=0)


class AmbiguityReference(BaseModel):
    reference: str
    reference_kind: Literal[
        "pronoun",
        "demonstrative",
        "relative",
    ]
    position: int = Field(ge=0)
    span: TextSpan
    linguistic: dict[str, str | int | None]
    candidate_count: int = Field(ge=0)
    is_ambiguous: bool
    signals: dict[str, bool]
    candidate_interpretations: list[AmbiguityCandidate] = []
    similarity_gap: float = Field(ge=0, le=1)
    best_candidate: str | None = None
    confidence: float = Field(ge=0, le=1)
    competition_score: float = Field(ge=0, le=1)


class LexicalSenseCandidate(BaseModel):
    sense: str
    definition: str
    similarity: float
    combined_score: float


class LexicalAmbiguity(BaseModel):
    term: str
    span: TextSpan
    candidate_senses: list[LexicalSenseCandidate] = []
    candidate_count: int = Field(ge=0)
    top2_margin: float | None = None
    normalized_entropy: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )


class AmbiguityResult(BaseModel):
    is_ambiguous: bool
    score: float = Field(
        ge=0,
        le=1,
        description="1 = highly ambiguous",
    )
    spans: list[TextSpan] = []
    references: list[AmbiguityReference] = []
    lexical_ambiguities: list[LexicalAmbiguity] = []


class VaguenessResult(BaseModel):
    count: int = Field(ge=0)
    terms: list[TextSpan] = []


class StabilityResult(BaseModel):
    label: Literal["stable", "potentially_unstable"]
    score: float = Field(
        ge=0,
        le=1,
        description="1 = interpretation fully stable under rewording",
    )


class RequirementAnalysis(BaseModel):
    requirement_id: str
    quality: QualityResult
    ambiguity: AmbiguityResult
    vagueness: VaguenessResult
    stability: StabilityResult
    confidence: float = Field(ge=0, le=1)
    explanation: str
    model_version: str
    analysed_at: datetime


class LexicalMeaningSelection(BaseModel):
    """A meaning explicitly selected by the user for one word."""

    term: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    definition: str = Field(min_length=1)


class RequirementRegenerationIn(BaseModel):
    """Original requirement and the user's selected lexical meanings."""

    text: str = Field(min_length=1)
    lexical_choices: list[LexicalMeaningSelection] = Field(
        default_factory=list
    )


class RequirementRegenerationOut(BaseModel):
    """The regenerated requirement returned by the AI service."""

    regenerated_text: str
