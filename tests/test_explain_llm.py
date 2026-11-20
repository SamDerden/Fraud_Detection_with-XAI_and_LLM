"""SHAP reconstruction and Grok response validation tests (no network needed)."""
import json

import pandas as pd
from catboost import CatBoostClassifier

from antifraud.explain import deterministic_explanation, explain_prediction
from antifraud.llm import explain_with_optional_grok
from antifraud.model_artifact import CATEGORICAL, MODEL_FEATURES, load_artifact


def _factors():
    return [
        {"feature": "tx_count_24h", "value": 8, "shap_log_odds": .5, "direction": "increases_risk"},
        {"feature": "device_age_days", "value": 100, "shap_log_odds": -.2, "direction": "decreases_risk"},
    ]


def test_shap_reconciles_with_raw_model_margin(tmp_path):
    data = []
    for i in range(30):
        row = {name: float(i + 1) for name in MODEL_FEATURES}
        row.update({"hour": i % 24, "day_of_week": i % 7,
                    "channel": "mobile", "country": "RU",
                    "is_new_device": int(i < 7), "is_night_tx": int(i % 24 <= 5)})
        data.append(row)
    x = pd.DataFrame(data, columns=list(MODEL_FEATURES))
    model = CatBoostClassifier(iterations=12, depth=2, verbose=False,
                               cat_features=list(CATEGORICAL), random_seed=42)
    model.fit(x, [0] * 15 + [1] * 15)
    model.save_model(str(tmp_path / "model.cbm"))
    (tmp_path / "metadata.json").write_text(json.dumps({
        "model_type": "CatBoostClassifier", "model_version": "test-shap",
        "features": list(MODEL_FEATURES),
    }))
    load_artifact.cache_clear()
    explained = explain_prediction(str(tmp_path), data[17])
    assert explained["explanation_method"] == "catboost_native_shap"
    assert 0 <= explained["risk_score"] <= 1
    assert 1 <= len(explained["top_factors"]) <= 5


def test_no_credentials_fallback(monkeypatch):
    monkeypatch.setenv("ANTIFRAUD_LLM_ENABLED", "true")
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    text, method = explain_with_optional_grok(_factors())
    assert method == "shap_template_no_credentials"
    assert "tx_count_24h" in text


def test_llm_rejects_invented_feature(monkeypatch):
    import antifraud.llm as llm

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": '{"factor_names":["secret_account"]}'}}]}

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            assert "user_id" not in json.dumps(kwargs.get("json", {}))
            return FakeResponse()

    monkeypatch.setenv("ANTIFRAUD_LLM_ENABLED", "true")
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    monkeypatch.setenv("GROK_MODEL", "test-model")
    monkeypatch.setattr(llm.httpx, "Client", FakeClient)
    text, method = llm.explain_with_optional_grok(_factors())
    assert method == "shap_template_fallback"
    assert "secret_account" not in text


def test_llm_accepts_only_grounded_factor_names(monkeypatch):
    import antifraud.llm as llm

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": '{"factor_names":["device_age_days"]}'}}]}

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setenv("ANTIFRAUD_LLM_ENABLED", "true")
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    monkeypatch.setenv("GROK_MODEL", "test-model")
    monkeypatch.setattr(llm.httpx, "Client", FakeClient)
    text, method = llm.explain_with_optional_grok(_factors())
    assert method == "grok_grounded_selection"
    assert "device_age_days" in text
    assert "tx_count_24h" not in text
