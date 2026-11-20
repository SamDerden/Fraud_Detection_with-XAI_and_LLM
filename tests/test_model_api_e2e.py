"""E2E smoke test: persisted CatBoost -> FastAPI, no private CSV required."""
import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from fastapi.testclient import TestClient

from antifraud.app import app
from antifraud.model_artifact import MODEL_FEATURES, CATEGORICAL, load_artifact


def test_persisted_model_served_via_api(tmp_path, monkeypatch):
    rows = []
    for i in range(40):
        rows.append({
            "amount": 100.0 + i * 200, "hour": i % 24,
            "day_of_week": i % 7, "avg_amount_30d": 300.0,
            "tx_count_24h": i % 8, "tx_count_7d": i % 20,
            "distinct_merchants_7d": 2, "distinct_countries_30d": 1,
            "device_age_days": i % 20, "is_new_device": int(i % 20 < 7),
            "is_night_tx": int(i % 24 <= 5),
            "channel": "mobile", "country": "RU",
        })
    x = pd.DataFrame(rows, columns=list(MODEL_FEATURES))
    y = np.array([int(i >= 20) for i in range(40)])
    model = CatBoostClassifier(iterations=10, depth=2, verbose=False,
                               random_seed=42, cat_features=list(CATEGORICAL))
    model.fit(x, y)
    model.save_model(str(tmp_path / "model.cbm"))
    (tmp_path / "metadata.json").write_text(json.dumps({
        "model_type": "CatBoostClassifier", "model_version": "smoke-v1",
        "features": list(MODEL_FEATURES), "review_threshold": 0.5,
    }))
    load_artifact.cache_clear()
    monkeypatch.setenv("ANTIFRAUD_MODEL_DIR", str(tmp_path))
    with TestClient(app) as client:
        info = client.get("/v1/model-info")
        assert info.status_code == 200
        assert info.json()["model_version"] == "smoke-v1"
        response = client.post("/v1/score", json={
            "transaction_id": "t-1", "user_id": "u-1",
            "amount": 1200, "timestamp": "2025-12-05T03:00:00Z",
            "hour": 3, "day_of_week": 4,
            "avg_amount_30d": 300, "tx_count_24h": 5,
            "tx_count_7d": 10, "distinct_merchants_7d": 2,
            "distinct_countries_30d": 1, "device_age_days": 0,
            "is_new_device": 1, "is_night_tx": 1,
            "channel": "mobile", "country": "RU",
        })
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["model_version"] == "smoke-v1"
        assert body["decision"] in ("approve", "review")
        assert body["demo_only"] is False
        assert body["explanation_method"] == "shap_template"
        assert len(body["top_factors"]) == 5
        assert "SHAP" in body["explanation"]
        assert 0 <= body["risk_score"] <= 1
