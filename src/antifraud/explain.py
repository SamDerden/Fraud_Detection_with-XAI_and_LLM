"""Explain a trusted CatBoost prediction using native SHAP contributions.

The feature contributions are in raw log-odds units; they are NOT probability
percentage points. The bias term + contributions equals the raw model margin.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from catboost import Pool

from antifraud.model_artifact import CATEGORICAL, MODEL_FEATURES, load_artifact


def explain_prediction(directory: str, features: dict, top_k: int = 5) -> dict:
    if not 1 <= top_k <= len(MODEL_FEATURES):
        raise ValueError("top_k outside valid range")
    model, meta = load_artifact(directory)
    row = pd.DataFrame([{name: features[name] for name in MODEL_FEATURES}])
    for col in CATEGORICAL:
        row[col] = row[col].astype(str)
    shap = model.get_feature_importance(
        Pool(row, cat_features=list(CATEGORICAL)), type="ShapValues"
    )
    if shap.shape != (1, len(MODEL_FEATURES) + 1):
        raise RuntimeError("Unexpected CatBoost SHAP output shape")
    values = shap[0, :-1]
    bias = float(shap[0, -1])
    raw_margin = float(model.predict(row, prediction_type="RawFormulaVal")[0])
    if not np.isclose(bias + float(values.sum()), raw_margin, atol=1e-5):
        raise RuntimeError("SHAP contributions do not reconcile with prediction")
    probability = 1.0 / (1.0 + math.exp(-raw_margin))
    if not math.isfinite(probability):
        raise RuntimeError("Nonfinite model score")
    ranked = sorted(zip(MODEL_FEATURES, values), key=lambda pair: abs(pair[1]), reverse=True)
    return {
        "risk_score": probability,
        "shap_base_log_odds": bias,
        "shap_total_log_odds": raw_margin,
        "top_factors": [
            {"feature": name, "value": features[name], "shap_log_odds": round(float(contribution), 6),
             "direction": "increases_risk" if contribution > 0 else "decreases_risk"}
            for name, contribution in ranked[:top_k]
        ],
        "explanation_method": "catboost_native_shap",
        "model_version": meta["model_version"],
    }


def deterministic_explanation(factors: list[dict]) -> str:
    if not factors:
        return "Модель рассчитала оценку риска, но объясняющие факторы недоступны."
    raised = [f["feature"] for f in factors if f["direction"] == "increases_risk"]
    lowered = [f["feature"] for f in factors if f["direction"] == "decreases_risk"]
    segments = []
    if raised:
        segments.append("Повышают оценку риска: " + ", ".join(raised))
    if lowered:
        segments.append("Снижают оценку риска: " + ", ".join(lowered))
    return ". ".join(segments) + ". Это диагностические факторы модели, а не доказательство мошенничества."
