def test_analyze_returns_the_contract_shape(client):
    response = client.post("/api/v1/analyze", json={"requirement_id": "REQ-1", "text": "The system shall be fast."})
    assert response.status_code == 200
    body = response.json()
    assert body["requirement_id"] == "REQ-1"
    assert {"quality", "ambiguity", "vagueness", "stability", "confidence", "explanation"} <= set(body)


def test_analyze_rejects_empty_text(client):
    response = client.post("/api/v1/analyze", json={"requirement_id": "REQ-1", "text": ""})
    assert response.status_code == 422


def test_batch_keeps_the_order_of_requirements(client):
    payload = [{"requirement_id": f"REQ-{i}", "text": "Some requirement"} for i in range(3)]
    response = client.post("/api/v1/analyze/batch", json=payload)
    assert response.status_code == 200
    assert [item["requirement_id"] for item in response.json()] == ["REQ-0", "REQ-1", "REQ-2"]
