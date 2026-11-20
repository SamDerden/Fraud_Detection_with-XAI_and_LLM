"""Behavioral, edge-case and contract tests for the educational model."""
import numpy as np
from fastapi.testclient import TestClient

from antifraud.app import app, ScoreRequest, build_features, get_model

client = TestClient(app)


def test_probabilities_are_valid():
    x = build_features(ScoreRequest(transaction_id="t", user_id="u",
        amount=1200, timestamp="2025-12-05T03:00:00Z"))
    p = get_model().predict_proba(x)
    assert p.shape == (1, 2)
    assert np.all((p >= 0) & (p <= 1))
    assert np.allclose(p.sum(axis=1), 1)


def test_large_amount_is_finite():
    x = build_features(ScoreRequest(transaction_id="t", user_id="u",
        amount=1e10, timestamp="2025-12-05T03:00:00Z"))
    assert np.isfinite(x).all()


def test_empty_id_rejected():
    r = client.post("/v1/score", json={"transaction_id": "", "user_id": "u",
        "amount": 100, "timestamp": "2025-12-05T03:00:00Z"})
    assert r.status_code == 422


def test_unsupported_currency_rejected():
    r = client.post("/v1/score", json={"transaction_id": "t", "user_id": "u",
        "amount": 100, "currency": "USD", "timestamp": "2025-12-05T03:00:00Z"})
    assert r.status_code == 422


def test_negative_history_count_rejected():
    r = client.post("/v1/score", json={"transaction_id": "t", "user_id": "u",
        "amount": 100, "tx_count_7d": -3, "timestamp": "2025-12-05T03:00:00Z"})
    assert r.status_code == 422
