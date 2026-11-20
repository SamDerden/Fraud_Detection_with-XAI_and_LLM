"""ExtraTrees v4: exact training-time preprocessing and local perturbation explanation.

Trusted local joblib archive only. Explanations are one-feature-at-a-time score
deltas against training means, NOT SHAP and NOT causal attributions.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

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


def score_and_explain_extratrees(directory: str, features: dict, top_k: int = 5):
    bundle, meta = load_extratrees(directory)
    row = transform(features, meta["encoded_columns"])
    scaled = bundle["scaler"].transform(row)
    model = bundle["model"]
    score = float(model.predict_proba(scaled)[0, 1])
    if not np.isfinite(score):
        raise RuntimeError("ExtraTrees returned nonfinite score")
    # Replace a single encoded column with its train mean (scaled zero).
    # Correlated columns and one-hot dummies make this a diagnostic, not SHAP.
    deltas = []
    for j, col in enumerate(meta["encoded_columns"]):
        modified = scaled.copy()
        modified[0, j] = 0.0
        baseline = float(model.predict_proba(modified)[0, 1])
        delta = score - baseline
        deltas.append((col, delta))
    ranked = sorted(deltas, key=lambda t: abs(t[1]), reverse=True)[:top_k]
    factors = [
        {"feature": col, "value": float(row.iloc[0][col]),
         "score_delta": round(float(delta), 6),
         "direction": "increases_risk" if delta > 0 else "decreases_risk"}
        for col, delta in ranked
    ]
    return score, {**meta, "review_threshold": REVIEW_THRESHOLD}, factors
