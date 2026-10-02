import pytest
from fastapi.testclient import TestClient

from app import db
from app.api.v1 import routes
from app.main import create_app


class FakeTask1Analyzer:
    """Lightweight Task 1 analyzer used by backend API tests."""

    def analyze(self, text: str) -> dict:
        if text == "The system shall display the user dashboard within 2 seconds.":
            return {
                "label": "Good",
                "score": 0.95,
                "model_label": "ok",
                "model_probabilities": {
                    "defect": 0.05,
                    "ok": 0.95,
                },
                "structural_completeness": {
                    "complete": True,
                    "score": 1.0,
                    "checks": {
                        "subject": True,
                        "requirement_modal": True,
                        "action": True,
                        "object_or_complement": True,
                    },
                    "missing_elements": [],
                },
                "issues": [],
            }

        if text == "The system should be good and fast for users.":
            return {
                "label": "Poor",
                "score": 0.25,
                "model_label": "defect",
                "model_probabilities": {
                    "defect": 0.75,
                    "ok": 0.25,
                },
                "structural_completeness": {
                    "complete": True,
                    "score": 1.0,
                    "checks": {
                        "subject": True,
                        "requirement_modal": True,
                        "action": True,
                        "object_or_complement": True,
                    },
                    "missing_elements": [],
                },
                "issues": [
                    {
                        "type": "quality_defect",
                        "message": (
                            "The trained QuRE classifier identified this "
                            "requirement as potentially defective."
                        ),
                    }
                ],
            }

        if text == "The system shall.":
            return {
                "label": "Needs Clarification",
                "score": 0.25,
                "model_label": "ok",
                "model_probabilities": {
                    "defect": 0.20,
                    "ok": 0.80,
                },
                "structural_completeness": {
                    "complete": False,
                    "score": 0.25,
                    "checks": {
                        "subject": True,
                        "requirement_modal": True,
                        "action": False,
                        "object_or_complement": False,
                    },
                    "missing_elements": [
                        "action",
                        "object or complement",
                    ],
                },
                "issues": [
                    {
                        "type": "missing_information",
                        "message": "Missing structural element: action.",
                    },
                    {
                        "type": "missing_information",
                        "message": (
                            "Missing structural element: object or complement."
                        ),
                    },
                ],
            }

        raise AssertionError(f"Unexpected test requirement: {text}")


@pytest.fixture
def client(monkeypatch):
    # Unit tests never need the real database.
    monkeypatch.setattr(db, "database_ok", lambda: False)

    # Backend API tests do not need the real 746 MB Task 1 model.
    monkeypatch.setattr(routes, "_task1_analyzer", FakeTask1Analyzer())

    return TestClient(create_app())
