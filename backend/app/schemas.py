"""Request and response models: the API contract of this service.

They mirror the JSON Schemas in Synapse-Web/contracts/requirement-quality; change both together.
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
    type: str = Field(description="Defect category, e.g. incomplete, inconsistent, grammar, missing_information")
    message: str
    span: TextSpan | None = None


class QualityResult(BaseModel):
    label: str = Field(description="Overall quality category")
    score: float = Field(ge=0, le=1, description="1 = high quality")
    issues: list[QualityIssue] = []


class AmbiguityResult(BaseModel):
    is_ambiguous: bool
    score: float = Field(ge=0, le=1, description="1 = highly ambiguous")
    spans: list[TextSpan] = []


class VaguenessResult(BaseModel):
    count: int = Field(ge=0)
    terms: list[TextSpan] = []


class StabilityResult(BaseModel):
    label: Literal["stable", "potentially_unstable"]
    score: float = Field(ge=0, le=1, description="1 = interpretation fully stable under rewording")


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
