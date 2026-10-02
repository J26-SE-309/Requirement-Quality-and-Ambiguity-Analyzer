"""Inference utilities for the Task 1 requirement-quality model."""

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from quality_ml.config import DATA_DIR

DEFAULT_MODEL_DIR = (
    DATA_DIR / "task1_exp06_all_layers" / "final_model"
)


class RequirementQualityModel:
    """Load the fine-tuned DeBERTa model and classify requirements."""

    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR):
        self.model_dir = Path(model_dir)

        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Task 1 model directory not found: {self.model_dir}"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir
        )

        self.model.eval()

    def predict(self, text: str) -> dict:
        """Predict defect/ok label and return class probabilities."""
        if not text or not text.strip():
            raise ValueError("Requirement text must not be empty.")

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )

        with torch.inference_mode():
            outputs = self.model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=-1)[0]
        predicted_id = int(torch.argmax(probabilities))

        id2label = {
            int(key): value
            for key, value in self.model.config.id2label.items()
        }

        return {
            "label": id2label[predicted_id],
            "score": float(probabilities[predicted_id]),
            "probabilities": {
                id2label[index]: float(probabilities[index])
                for index in range(len(probabilities))
            },
        }