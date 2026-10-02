"""Task 1 contextual requirement quality assessment."""

from quality_ml.inference import RequirementQualityModel
from quality_ml.quality_checks import check_structural_completeness


class Task1QualityAnalyzer:
    """Combine DeBERTa classification with spaCy structural checks."""

    def __init__(self):
        self.model = RequirementQualityModel()

    def analyze(self, text: str) -> dict:
        """Assess requirement quality using the trained model and structural checks."""
        if not text or not text.strip():
            raise ValueError("Requirement text must not be empty.")

        model_result = self.model.predict(text)
        structure_result = check_structural_completeness(text)

        model_label = model_result["label"]
        ok_probability = model_result["probabilities"]["ok"]

        if not structure_result["complete"]:
            quality_label = "Needs Clarification"
        elif model_label == "ok":
            quality_label = "Good"
        else:
            quality_label = "Poor"

        # quality.score follows the API contract:
        # 1.0 = high quality, 0.0 = low quality.
        #
        # The DeBERTa OK probability represents the learned quality signal.
        # Structural completeness is used as a second quality signal.
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

        if model_label == "defect":
            issues.append(
                {
                    "type": "quality_defect",
                    "message": (
                        "The trained QuRE classifier identified this "
                        "requirement as potentially defective."
                    ),
                }
            )

        return {
            "label": quality_label,
            "score": quality_score,
            "model_label": model_label,
            "model_probabilities": model_result["probabilities"],
            "structural_completeness": structure_result,
            "issues": issues,
        }