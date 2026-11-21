"""ExtraTrees v4: exact training-time preprocessing and TreeSHAP explanations.

Trusted local joblib archive only. Explanations are additive TreeSHAP
contributions in probability units, NOT causal attributions.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

BASE = ("amount", "hour", "day_of_week", "avg_amount_30d", "tx_count_24h",
        "tx_count_7d", "distinct_merchants_7d", "distinct_countries_30d",
        "device_age_days", "is_new_device", "is_night_tx", "channel", "country")
ENGINEERED = ("amount_to_avg", "log_amount", "log_avg_amount", "tx24_share",
              "merchants_per_tx", "tx_per_day", "log_device_age")
VERSION = "extratrees-engineered-v4"
# Selected on v3 validation with FPR <= 10%; educational review routing only.
REVIEW_THRESHOLD = 0.049247396161697266


@lru_cache(maxsize=4)
def load_extratrees(directory: str):
    root = Path(directory).resolve()
    meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if (meta.get("model_version") != VERSION
            or meta.get("model_type") != "ExtraTreesClassifier"
            or tuple(meta.get("base_features", [])) != BASE
            or tuple(meta.get("engineered_features", [])) != ENGINEERED):
        raise ValueError("ExtraTrees v4 metadata mismatch")
    bundle = joblib.load(root / "model.joblib")
    columns = tuple(meta["encoded_columns"])
    if tuple(bundle["encoded_columns"]) != columns:
        raise ValueError("ExtraTrees v4 encoded columns mismatch")
    if bundle["scaler"].n_features_in_ != len(columns):
        raise ValueError("ExtraTrees v4 scaler dimension mismatch")
    if bundle["model"].n_features_in_ != len(columns):
        raise ValueError("ExtraTrees v4 model dimension mismatch")
    return bundle, {**meta, 'review_threshold': REVIEW_THRESHOLD}


def transform(features: dict, columns: list[str]):
    x = pd.DataFrame([{k: features[k] for k in BASE}])
    x["amount_to_avg"] = x.amount / (x.avg_amount_30d + 1)
    x["log_amount"] = np.log1p(x.amount)
    x["log_avg_amount"] = np.log1p(x.avg_amount_30d)
    x["tx24_share"] = x.tx_count_24h / (x.tx_count_7d + 1)
    x["merchants_per_tx"] = x.distinct_merchants_7d / (x.tx_count_7d + 1)
    x["tx_per_day"] = x.tx_count_7d / 7
    x["log_device_age"] = np.log1p(x.device_age_days)
    x = pd.get_dummies(x, columns=["channel", "country"], dtype=int)
    return x.reindex(columns=columns, fill_value=0)


# Aggregate mutually related encoded features into readable concepts. Grouping is
# an exact sum of SHAP contributions, not a causal interpretation.
def _feature_group(name: str) -> str:
    if name.startswith("channel_"):
        return "channel"
    if name.startswith("country_"):
        return "country"
    if name in ("device_age_days", "log_device_age", "is_new_device"):
        return "device_history"
    if name in ("amount", "log_amount", "avg_amount_30d", "log_avg_amount", "amount_to_avg"):
        return "amount_context"
    if name in ("tx_count_24h", "tx_count_7d", "tx24_share", "tx_per_day"):
        return "transaction_frequency"
    if name in ("distinct_merchants_7d", "merchants_per_tx"):
        return "merchant_diversity"
    return name


@lru_cache(maxsize=4)
def _tree_explainer(directory: str):
    bundle, _ = load_extratrees(directory)
    return shap.TreeExplainer(bundle["model"])


def _positive_class_shap(explainer, scaled):
    result = explainer.shap_values(scaled, check_additivity=False)
    if isinstance(result, list):
        values = np.asarray(result[1], dtype=float)[0]
    else:
        values = np.asarray(result, dtype=float)
        if values.ndim == 3:
            values = values[0, :, 1]
        elif values.ndim == 2:
            values = values[0]
        else:
            raise RuntimeError("Unexpected SHAP value dimensions")
    expected = np.asarray(explainer.expected_value, dtype=float).reshape(-1)
    base = float(expected[1] if len(expected) == 2 else expected[0])
    return np.asarray(values, dtype=float).reshape(-1), base


def score_and_explain_extratrees(directory: str, features: dict, top_k: int = 5):
    bundle, meta = load_extratrees(directory)
    row = transform(features, meta["encoded_columns"])
    scaled = bundle["scaler"].transform(row)
    score = float(bundle["model"].predict_proba(scaled)[0, 1])
    if not np.isfinite(score):
        raise RuntimeError("ExtraTrees returned nonfinite score")
    shap_values, base = _positive_class_shap(_tree_explainer(directory), scaled)
    if len(shap_values) != len(meta["encoded_columns"]) or not np.isfinite(shap_values).all():
        raise RuntimeError("SHAP dimensions or values invalid")
    # TreeSHAP for sklearn ExtraTreesClassifier is in class probability units.
    if not np.isclose(base + float(shap_values.sum()), score, atol=1e-5, rtol=1e-5):
        raise RuntimeError("SHAP contributions do not reconcile with model prediction")
    grouped = {}
    for col, contribution in zip(meta["encoded_columns"], shap_values):
        name = _feature_group(col)
        grouped[name] = grouped.get(name, 0.0) + float(contribution)
    if not np.isclose(base + sum(grouped.values()), score, atol=1e-5):
        raise RuntimeError("Grouped SHAP contributions do not reconcile")
    ranked_all = sorted(grouped.items(), key=lambda item: abs(item[1]), reverse=True)
    ranked = ranked_all[:top_k]
    remaining = float(sum(value for _, value in ranked_all[top_k:]))
    factors = [
        {"feature": name, "shap_value": round(value, 8),
         "direction": "increases_risk" if value > 0 else "decreases_risk"}
        for name, value in ranked
    ]
    return score, {**meta, "review_threshold": REVIEW_THRESHOLD,
                   "shap_other_value": remaining, "shap_all_factors": [
                       {"feature": name, "shap_value": float(value)} for name, value in ranked_all],
                   "shap_base_value": base, "shap_reconstruction": base + float(shap_values.sum()),
                   "explanation_type": "tree_shap_probability"}, factors
