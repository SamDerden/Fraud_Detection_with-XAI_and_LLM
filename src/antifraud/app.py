"""Demonstration-only antifraud API recovered from Colab seminars 9 and 11/12.

Synthetic training labels are generated from an explicit toy risk rule. This is NOT
a calibrated fraud probability or a deployable banking decision engine.
"""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from typing import Literal

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from sklearn.ensemble import RandomForestClassifier

FEATURE_NAMES = (
    "amount_log", "hour", "day_of_week", "tx_count_24h", "tx_count_7d",
    "distinct_merchants_7d", "distinct_countries_30d", "device_age_days",
    "is_new_device", "is_night_tx", "avg_amount_ratio", "channel_mobile",
)
MODEL_VERSION = "synthetic-demo-v1"
app = FastAPI(title="Antifraud synthetic demo", version="0.1.0")


class ScoreRequest(BaseModel):
    transaction_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    amount: float = Field(gt=0, allow_inf_nan=False)
    timestamp: datetime
    currency: str = "RUB"
    hour: int | None = Field(default=None, ge=0, le=23)
    day_of_week: int | None = Field(default=None, ge=0, le=6)
    tx_count_24h: int = Field(default=0, ge=0)
    tx_count_7d: int = Field(default=0, ge=0)
    distinct_merchants_7d: int = Field(default=1, ge=1)
    distinct_countries_30d: int = Field(default=1, ge=1)
    device_age_days: int = Field(default=100, ge=0)
    avg_amount_ratio: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    channel: Literal["mobile", "web", "pos", "api"] = "mobile"

    @field_validator("timestamp")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must include timezone")
        return value


class ScoreResponse(BaseModel):
    risk_score: float
    decision: Literal["approve", "review", "block"]
    explanation: str
    model_version: str
    transaction_id: str
    demo_only: bool = True


def build_features(req: ScoreRequest) -> np.ndarray:
    hour = req.hour if req.hour is not None else req.timestamp.hour
    dow = req.day_of_week if req.day_of_week is not None else req.timestamp.weekday()
    return np.array([[
        np.log1p(req.amount), hour, dow, req.tx_count_24h, req.tx_count_7d,
        req.distinct_merchants_7d, req.distinct_countries_30d,
        req.device_age_days, int(req.device_age_days < 7), int(hour <= 5),
        req.avg_amount_ratio, int(req.channel == "mobile"),
    ]], dtype=float)


def synthetic_training_data(n: int = 3000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate coherent synthetic features and toy labels, not real fraud ground truth."""
    rng = np.random.default_rng(seed)
    amount = rng.lognormal(8, 1.2, n)
    hour = rng.integers(0, 24, n)
    dow = rng.integers(0, 7, n)
    tx24 = rng.poisson(3, n)
    tx7 = tx24 + rng.poisson(10, n)
    merchants = rng.integers(1, 10, n)
    countries = rng.choice([1, 2, 3, 4], n, p=[0.7, 0.2, 0.08, 0.02])
    device_age = rng.integers(0, 400, n)
    ratio = rng.lognormal(0, 0.7, n)
    mobile = rng.integers(0, 2, n)
    x = np.column_stack([
        np.log1p(amount), hour, dow, tx24, tx7, merchants, countries,
        device_age, (device_age < 7).astype(int), (hour <= 5).astype(int),
        ratio, mobile,
    ])
    risk = (
        1.0 * (amount > 15000) + 1.2 * (tx24 >= 8)
        + 1.2 * (device_age < 7) + 0.9 * (countries >= 3)
        + 0.6 * (hour <= 5) + 0.8 * (ratio >= 3)
    )
    y = (risk + rng.normal(0, 0.7, n) >= 2.5).astype(int)
    return x, y


@lru_cache(maxsize=1)
def get_model() -> RandomForestClassifier:
    x, y = synthetic_training_data()
    model = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                   random_state=42, n_jobs=1)
    model.fit(x, y)
    return model


def decision_for(score: float) -> Literal["approve", "review", "block"]:
    if score >= 0.75:
        return "block"
    if score >= 0.45:
        return "review"
    return "approve"


@app.get("/health")
def health() -> dict:
    get_model()
    return {"status": "ok", "model_version": MODEL_VERSION, "demo_only": True,
            "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/v1/model-info")
def model_info() -> dict:
    return {"model_version": MODEL_VERSION, "model_type": "RandomForestClassifier",
            "n_features": len(FEATURE_NAMES), "feature_names": list(FEATURE_NAMES),
            "training_data": "synthetic", "demo_only": True}


@app.post("/v1/score", response_model=ScoreResponse)
def score_transaction(req: ScoreRequest) -> ScoreResponse:
    score = float(get_model().predict_proba(build_features(req))[0, 1])
    decision = decision_for(score)
    return ScoreResponse(
        risk_score=round(score, 4), decision=decision,
        explanation="Synthetic model demonstration; not SHAP or a real fraud assessment",
        model_version=MODEL_VERSION, transaction_id=req.transaction_id,
    )
