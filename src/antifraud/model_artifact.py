"""Load a trusted, locally generated CatBoost artifact and its validated metadata."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier

MODEL_FEATURES = (
    "amount", "hour", "day_of_week", "avg_amount_30d",
    "tx_count_24h", "tx_count_7d", "distinct_merchants_7d",
    "distinct_countries_30d", "device_age_days", "is_new_device",
    "is_night_tx", "channel", "country",
)
CATEGORICAL = ("channel", "country")


@lru_cache(maxsize=4)
def load_artifact(directory: str):
    root = Path(directory).resolve()
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if tuple(metadata["features"]) != MODEL_FEATURES:
        raise ValueError("Model feature schema mismatch")
    if metadata["model_type"] != "CatBoostClassifier":
        raise ValueError("Unexpected model type")
    model = CatBoostClassifier()
    model.load_model(str(root / "model.cbm"))
    return model, metadata


def predict_from_artifact(directory: str, features: dict):
    model, meta = load_artifact(directory)
    row = pd.DataFrame([{name: features[name] for name in MODEL_FEATURES}])
    for col in CATEGORICAL:
        row[col] = row[col].astype(str)
    return float(model.predict_proba(row)[0, 1]), meta
