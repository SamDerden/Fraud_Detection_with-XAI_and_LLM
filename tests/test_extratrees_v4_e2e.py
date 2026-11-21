"""Contract tests for the published ExtraTrees v4 artifact (no synthetic fallback)."""
import json
from pathlib import Path
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "extratrees-v4-artifact.zip"


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    from antifraud.extratrees_v4 import load_extratrees
    if not ARCHIVE.is_file():
        pytest.skip("ExtraTrees v4 ZIP is not present")
    root = tmp_path_factory.mktemp("extratrees-v4")
    with zipfile.ZipFile(ARCHIVE) as z:
        for name in ("model.joblib", "metadata.json"):
            (root / name).write_bytes(z.read("extratrees-v4/" + name))
    _, meta = load_extratrees(str(root))
    assert meta["model_version"] == "extratrees-engineered-v4"
    return root


def sample():
    return {
        "amount": 45000.0, "hour": 22, "day_of_week": 5,
        "avg_amount_30d": 5000.0, "tx_count_24h": 8,
        "tx_count_7d": 15, "distinct_merchants_7d": 5,
        "distinct_countries_30d": 2, "device_age_days": 2,
        "is_new_device": 1, "is_night_tx": 0,
        "channel": "mobile", "country": "RU",
    }


def test_v4_scoring_is_repeatable(model_dir):
    from antifraud.extratrees_v4 import score_and_explain_extratrees
    a = score_and_explain_extratrees(str(model_dir), sample())
    b = score_and_explain_extratrees(str(model_dir), sample())
    assert a == b
    score, meta, factors = a
    assert 0 <= score <= 1
    assert meta["review_threshold"] > 0
    assert len(factors) == 5
    assert all("shap_value" in f for f in factors)


def test_v4_api_end_to_end(model_dir, monkeypatch):
    from fastapi.testclient import TestClient
    from antifraud.app import app
    monkeypatch.setenv("ANTIFRAUD_MODEL_DIR", str(model_dir))
    monkeypatch.setenv("ANTIFRAUD_MODEL_KIND", "extratrees_v4")
    monkeypatch.setenv("ANTIFRAUD_REQUIRE_ARTIFACT", "1")
    monkeypatch.delenv("ANTIFRAUD_LLM_ENABLED", raising=False)
    client = TestClient(app)
    info = client.get("/v1/model-info")
    assert info.status_code == 200
    assert info.json()["model_version"] == "extratrees-engineered-v4"
    assert info.json()["n_engineered_features"] == 7
    req = {
        "transaction_id": "test-v4", "user_id": "demo",
        "timestamp": "2026-11-21T22:00:00Z", "amount": 45000,
        "avg_amount_30d": 5000, "tx_count_24h": 8, "tx_count_7d": 15,
        "distinct_merchants_7d": 5, "distinct_countries_30d": 2,
        "device_age_days": 2, "is_new_device": 1, "is_night_tx": 0,
        "channel": "mobile", "country": "RU",
    }
    response = client.post("/v1/score", json=req)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["model_version"] == "extratrees-engineered-v4"
    assert data["decision"] in ("approve", "review")
    assert len(data["top_factors"]) == 5
    assert "shap_value" in data["top_factors"][0]
    assert data["shap_base_value"] is not None
    assert data["shap_all_factors"]
    assert abs(data["shap_base_value"] + sum(f["shap_value"] for f in data["shap_all_factors"]) - data["risk_score"]) < 0.00015

    assert client.get("/health").json()["model_version"] == "extratrees-engineered-v4"
