def test_analyze_returns_good_for_complete_requirement(client):
    response = client.post(
        "/api/v1/analyze",
        json={
            "requirement_id": "REQ-1",
            "text": "The system shall display the user dashboard within 2 seconds.",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["requirement_id"] == "REQ-1"
    assert body["quality"]["label"] == "Good"
    assert 0 <= body["quality"]["score"] <= 1
    assert body["quality"]["issues"] == []
    assert body["model_version"] == "task1-exp06-deberta-spacy+task2-exp04-minilm"


def test_analyze_returns_poor_for_defective_requirement(client):
    response = client.post(
        "/api/v1/analyze",
        json={
            "requirement_id": "REQ-2",
            "text": "The system should be good and fast for users.",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["quality"]["label"] == "Poor"
    assert 0 <= body["quality"]["score"] <= 1
    assert any(
        issue["type"] == "quality_defect"
        for issue in body["quality"]["issues"]
    )


def test_analyze_returns_needs_clarification_for_incomplete_requirement(client):
    response = client.post(
        "/api/v1/analyze",
        json={
            "requirement_id": "REQ-3",
            "text": "The system shall.",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["quality"]["label"] == "Needs Clarification"
    assert 0 <= body["quality"]["score"] <= 1

    issue_types = {
        issue["type"]
        for issue in body["quality"]["issues"]
    }

    assert "missing_information" in issue_types


def test_analyze_rejects_empty_text(client):
    response = client.post(
        "/api/v1/analyze",
        json={"requirement_id": "REQ-1", "text": ""},
    )

    assert response.status_code == 422


def test_batch_keeps_the_order_of_requirements(client):
    payload = [
        {
            "requirement_id": "REQ-1",
            "text": "The system shall display the user dashboard within 2 seconds.",
        },
        {
            "requirement_id": "REQ-2",
            "text": "The system should be good and fast for users.",
        },
        {
            "requirement_id": "REQ-3",
            "text": "The system shall.",
        },
    ]

    response = client.post("/api/v1/analyze/batch", json=payload)

    assert response.status_code == 200

    body = response.json()

    assert [item["requirement_id"] for item in body] == [
        "REQ-1",
        "REQ-2",
        "REQ-3",
    ]

    assert [item["quality"]["label"] for item in body] == [
        "Good",
        "Poor",
        "Needs Clarification",
    ]
