"""Task 1 contextual requirement quality assessment."""

from quality_ml.inference import RequirementQualityModel
from quality_ml.quality_checks import (
    check_structural_completeness,
    find_vague_terms,
)


class Task1QualityAnalyzer:
    """Combine DeBERTa classification with spaCy quality checks."""

    def __init__(self):
        self.model = RequirementQualityModel()

    def analyze(self, text: str) -> dict:
        """Assess requirement quality using the trained model and quality checks."""

        if not text or not text.strip():
            raise ValueError("Requirement text must not be empty.")

        model_result = self.model.predict(text)
        structure_result = check_structural_completeness(text)
        vague_terms = find_vague_terms(text)

        model_label = model_result["label"]
        ok_probability = model_result["probabilities"]["ok"]

        if not structure_result["complete"]:
            quality_label = "Needs Clarification"
        elif model_label == "ok":
            quality_label = "Good"
        else:
            quality_label = "Poor"

        quality_score = min(
            ok_probability,
            structure_result["score"],
        )

        issues = []

        for element in structure_result["missing_elements"]:
            issues.append(
                {
                    "type": "missing_information",
                    "message": f"Missing structural element: {element}.",
                }
            )

        if vague_terms:
            issues.append(
                {
                    "type": "vague_terms",
                    "message": (
                        "Potentially vague terms detected: "
                        + ", ".join(f'"{term}"' for term in vague_terms)
                        + "."
                    ),
                    "terms": vague_terms,
                }
            )

        if model_label == "defect":
            issues.append(
                {
                    "type": "quality_defect",
                    "message": (
                        "The requirement may contain quality issues. "
                        "Review it for clearer and more measurable wording."
                    ),
                }
            )

        return {
            "label": quality_label,
            "score": quality_score,
            "model_label": model_label,
            "model_probabilities": model_result["probabilities"],
            "structural_completeness": structure_result,
            "vague_terms": vague_terms,
            "issues": issues,
        }
