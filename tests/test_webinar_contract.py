"""Webinar 9 API aliases and v3 history feature contract."""
from fastapi.testclient import TestClient
from antifraud.app import app


def test_webinar_routes_present():
    paths = app.openapi()["paths"]
    assert {"/predict", "/healthcheck", "/model-info", "/v1/score"}.issubset(paths)


def test_zero_history_and_float_device_age_are_valid():
    from antifraud.app import ScoreRequest
    req = ScoreRequest(transaction_id="tx", user_id="u", amount=100.0,
                       timestamp="2026-02-15T00:20:32+00:00", device_age_days=41.4845,
                       tx_count_7d=0, tx_count_24h=0, distinct_merchants_7d=0,
                       distinct_countries_30d=0)
    assert req.device_age_days == 41.4845
    assert req.distinct_merchants_7d == 0


def test_inconsistent_history_is_rejected_before_model(monkeypatch):
    monkeypatch.setenv("ANTIFRAUD_MODEL_DIR", "/fake/artifact")
    payload = dict(transaction_id="tx", user_id="u", amount=100.0,
                   timestamp="2026-02-15T00:20:32+00:00", hour=0, day_of_week=6,
                   device_age_days=41.4845, is_new_device=0, is_night_tx=1,
                   avg_amount_30d=100.0, tx_count_24h=2, tx_count_7d=1,
                   distinct_merchants_7d=1, distinct_countries_30d=0)
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "history" in response.json()["detail"].lower()
