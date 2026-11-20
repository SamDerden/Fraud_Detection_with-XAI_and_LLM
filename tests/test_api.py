from fastapi.testclient import TestClient
from antifraud.app import app, build_features, ScoreRequest, decision_for, synthetic_training_data

client = TestClient(app)
VALID = {
    "transaction_id": "txn-1", "user_id": "user-1", "amount": 12000,
    "timestamp": "2026-09-14T03:32:00Z", "device_age_days": 0,
    "tx_count_24h": 10,
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["demo_only"] is True


def test_model_info():
    response = client.get("/v1/model-info")
    assert response.status_code == 200
    assert response.json()["n_features"] == 12
    assert response.json()["training_data"] == "synthetic"


def test_scoring_contract():
    response = client.post("/v1/score", json=VALID)
    assert response.status_code == 200, response.text
    body = response.json()
    assert 0 <= body["risk_score"] <= 1
    assert body["decision"] in ("approve", "review", "block")
    assert body["transaction_id"] == VALID["transaction_id"]
    assert body["demo_only"] is True


def test_invalid_amount_rejected():
    response = client.post("/v1/score", json={**VALID, "amount": -1})
    assert response.status_code == 422


def test_invalid_timestamp_rejected():
    response = client.post("/v1/score", json={**VALID, "timestamp": "bad"})
    assert response.status_code == 422


def test_naive_timestamp_rejected():
    response = client.post("/v1/score", json={**VALID, "timestamp": "2026-09-14T03:32:00"})
    assert response.status_code == 422


def test_zero_device_age_preserved():
    req = ScoreRequest.model_validate(VALID)
    features = build_features(req)
    assert features.shape == (1, 12)
    assert features[0, 7] == 0
    assert features[0, 8] == 1


def test_decision_boundaries():
    assert decision_for(0.4499) == "approve"
    assert decision_for(0.45) == "review"
    assert decision_for(0.75) == "block"


def test_synthetic_training_shape():
    x, y = synthetic_training_data(100)
    assert x.shape == (100, 12)
    assert set(y).issubset({0, 1})
