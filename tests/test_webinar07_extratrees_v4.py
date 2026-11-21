"""Six real ExtraTrees v4 checks for webinar 07; requires committed model artifact."""
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from antifraud.extratrees_v4 import BASE, ENGINEERED, VERSION, load_extratrees, score_and_explain_extratrees, transform
from antifraud.app import app

ARCHIVE = Path(__file__).resolve().parents[1] / "extratrees-v4-artifact.zip"

@pytest.fixture(scope="module")
def final_model(tmp_path_factory):
    assert ARCHIVE.is_file(), "Required ExtraTrees v4 archive is missing"
    root = tmp_path_factory.mktemp("webinar07-v4")
    with zipfile.ZipFile(ARCHIVE) as archive:
        for filename in ("model.joblib", "metadata.json"):
            (root / filename).write_bytes(archive.read("extratrees-v4/" + filename))
    bundle, meta = load_extratrees(str(root))
    return root, bundle, meta

@pytest.fixture
def operation():
    return dict(amount=850.0, hour=13, day_of_week=5, avg_amount_30d=1700.0,
                tx_count_24h=1, tx_count_7d=5, distinct_merchants_7d=3,
                distinct_countries_30d=1, device_age_days=220, is_new_device=0,
                is_night_tx=0, channel="mobile", country="RU")

def test_01_functionality_v4(final_model, operation):
    root, _, meta = final_model
    score, _, factors = score_and_explain_extratrees(str(root), operation)
    assert meta["model_version"] == VERSION
    assert np.isfinite(score) and 0 <= score <= 1
    assert len(factors) == 5

def test_02_model_schema_and_parameters(final_model):
    _, bundle, meta = final_model
    model = bundle["model"]
    assert len(BASE) == 13 and len(ENGINEERED) == 7
    assert model.n_estimators == 200 and model.max_depth == 4 and model.min_samples_leaf == 6
    assert tuple(meta["base_features"]) == BASE and tuple(meta["engineered_features"]) == ENGINEERED

def test_03_repeatability_v4(final_model, operation):
    root, _, _ = final_model
    first = score_and_explain_extratrees(str(root), operation)
    second = score_and_explain_extratrees(str(root), operation)
    assert first[0] == pytest.approx(second[0], abs=1e-12)
    assert first[1] == second[1]
    assert len(first[2]) == len(second[2])
    for a, b in zip(first[2], second[2]):
        assert a["feature"] == b["feature"]
        assert a["direction"] == b["direction"]
        assert a["shap_value"] == pytest.approx(b["shap_value"], abs=1e-8)

def test_04_engineered_features_finite(final_model, operation):
    _, bundle, meta = final_model
    row = transform(operation, meta["encoded_columns"])
    assert list(row.columns) == meta["encoded_columns"]
    assert np.isfinite(row.to_numpy(dtype=float)).all()
    assert np.isfinite(bundle["scaler"].transform(row)).all()

def test_05_history_and_device_contract_v4(final_model, monkeypatch):
    root, _, _ = final_model
    monkeypatch.setenv("ANTIFRAUD_MODEL_DIR", str(root))
    monkeypatch.setenv("ANTIFRAUD_MODEL_KIND", "extratrees_v4")
    monkeypatch.setenv("ANTIFRAUD_REQUIRE_ARTIFACT", "1")
    monkeypatch.setenv("ANTIFRAUD_LLM_ENABLED", "false")
    payload=dict(transaction_id="v4-w7",user_id="demo",amount=850,
                 timestamp="2026-11-21T13:00:00Z",avg_amount_30d=1700,
                 tx_count_24h=6,tx_count_7d=5,distinct_merchants_7d=3,
                 distinct_countries_30d=1,device_age_days=220,is_new_device=0,
                 is_night_tx=0,channel="mobile",country="RU")
    with TestClient(app) as client:
        response=client.post("/v1/score",json=payload)
    assert response.status_code == 422, response.text

def test_06_api_v4_response_and_metrics(final_model, monkeypatch):
    root, _, _ = final_model
    monkeypatch.setenv("ANTIFRAUD_MODEL_DIR", str(root))
    monkeypatch.setenv("ANTIFRAUD_MODEL_KIND", "extratrees_v4")
    monkeypatch.setenv("ANTIFRAUD_REQUIRE_ARTIFACT", "1")
    monkeypatch.setenv("ANTIFRAUD_LLM_ENABLED", "false")
    payload=dict(transaction_id="v4-w7",user_id="demo",amount=850,
                 timestamp="2026-11-21T13:00:00Z",avg_amount_30d=1700,
                 tx_count_24h=1,tx_count_7d=5,distinct_merchants_7d=3,
                 distinct_countries_30d=1,device_age_days=220,is_new_device=0,
                 is_night_tx=0,channel="mobile",country="RU")
    with TestClient(app) as client:
        response=client.post("/v1/score",json=payload)
    assert response.status_code == 200, response.text
    body=response.json()
    assert body["model_version"] == VERSION
    assert body["decision"] in ("approve","review")
    assert 0 <= body["risk_score"] <= 1
    assert len(body["top_factors"]) == 5
    assert all(np.isfinite(f["shap_value"]) for f in body["top_factors"])
