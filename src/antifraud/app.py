"""Demonstration-only antifraud API recovered from Colab seminars 9 and 11/12.

Synthetic training labels are generated from an explicit toy risk rule. This is NOT
a calibrated fraud probability or a deployable banking decision engine.
"""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import os
from typing import Literal

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from sklearn.ensemble import RandomForestClassifier

from antifraud.observability import install_observability
from antifraud.model_artifact import predict_from_artifact, load_artifact
from antifraud.explain import explain_prediction
from antifraud.llm import explain_with_optional_grok

FEATURE_NAMES = (
    "amount_log", "hour", "day_of_week", "tx_count_24h", "tx_count_7d",
    "distinct_merchants_7d", "distinct_countries_30d", "device_age_days",
    "is_new_device", "is_night_tx", "avg_amount_ratio", "channel_mobile",
)
MODEL_VERSION = "synthetic-demo-v1"
app = FastAPI(title="Antifraud synthetic demo", version="0.1.0")
install_observability(app)


class ScoreRequest(BaseModel):
    transaction_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    amount: float = Field(gt=0, allow_inf_nan=False)
    timestamp: datetime
    currency: Literal["RUB"] = "RUB"
    country: str = Field(default="RU", min_length=2, max_length=2)
    avg_amount_30d: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    is_new_device: int | None = Field(default=None, ge=0, le=1)
    is_night_tx: int | None = Field(default=None, ge=0, le=1)
    hour: int | None = Field(default=None, ge=0, le=23)
    day_of_week: int | None = Field(default=None, ge=0, le=6)
    tx_count_24h: int = Field(default=0, ge=0)
    tx_count_7d: int = Field(default=0, ge=0)
    distinct_merchants_7d: int = Field(default=0, ge=0)
    distinct_countries_30d: int = Field(default=0, ge=0)
    device_age_days: float = Field(default=100, ge=0, allow_inf_nan=False)
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
    explanation_method: str | None = None
    top_factors: list[dict] | None = None


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
    artifact = os.getenv("ANTIFRAUD_MODEL_DIR")
    if artifact:
        _, meta = load_artifact(artifact)
        version, demo = meta["model_version"], False
    else:
        get_model()
        version, demo = MODEL_VERSION, True
    return {"status": "ok", "model_version": version, "demo_only": demo,
            "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/v1/model-info")
def model_info() -> dict:
    artifact = os.getenv("ANTIFRAUD_MODEL_DIR")
    if artifact:
        _, meta = load_artifact(artifact)
        return {"model_version": meta["model_version"], "model_type": meta["model_type"],
                "n_features": len(meta["features"]), "feature_names": meta["features"],
                "training_data": "local_csv", "demo_only": False,
                "review_threshold": meta["review_threshold"], "block_threshold": None}
    return {"model_version": MODEL_VERSION, "model_type": "RandomForestClassifier",
            "n_features": len(FEATURE_NAMES), "feature_names": list(FEATURE_NAMES),
            "training_data": "synthetic", "demo_only": True}


@app.post("/v1/score", response_model=ScoreResponse)
def score_transaction(req: ScoreRequest) -> ScoreResponse:
    artifact = os.getenv("ANTIFRAUD_MODEL_DIR")
    if artifact:
        # No implicit defaults for engineered features: the caller must supply them.
        if req.avg_amount_30d is None or req.is_new_device is None or req.is_night_tx is None:
            from fastapi import HTTPException
            raise HTTPException(status_code=422, detail="Missing required engineered features for artifact model")
        hour = req.hour if req.hour is not None else req.timestamp.hour
        weekday = req.day_of_week if req.day_of_week is not None else req.timestamp.weekday()
        utc_timestamp = req.timestamp.astimezone(timezone.utc)
        if (hour != utc_timestamp.hour or weekday != utc_timestamp.weekday()
                or req.is_new_device != int(req.device_age_days < 7)
                or req.is_night_tx != int(hour <= 5)):
            from fastapi import HTTPException
            raise HTTPException(status_code=422, detail="Inconsistent derived features")
        if req.tx_count_7d < req.tx_count_24h or req.distinct_merchants_7d > req.tx_count_7d:\n            from fastapi import HTTPException\n            raise HTTPException(status_code=422, detail="Inconsistent transaction history counts")\n        values = {
            "amount": req.amount, "hour": hour, "day_of_week": weekday,
            "avg_amount_30d": req.avg_amount_30d, "tx_count_24h": req.tx_count_24h,
            "tx_count_7d": req.tx_count_7d,
            "distinct_merchants_7d": req.distinct_merchants_7d,
            "distinct_countries_30d": req.distinct_countries_30d,
            "device_age_days": req.device_age_days, "is_new_device": req.is_new_device,
            "is_night_tx": req.is_night_tx, "channel": req.channel,
            "country": req.country,
        }
        score, meta = predict_from_artifact(artifact, values)
        shap = explain_prediction(artifact, values)
        explanation, explanation_method = explain_with_optional_grok(shap["top_factors"])
        # A score from synthetic labels is NOT a calibrated fraud probability.
        # Never automatically block based on this educational model.
        decision = "review" if score >= meta["review_threshold"] else "approve"
        return ScoreResponse(
            risk_score=round(score, 4), decision=decision,
            explanation=explanation,
            model_version=meta["model_version"], transaction_id=req.transaction_id,
            demo_only=False, explanation_method=explanation_method,
            top_factors=shap["top_factors"],
        )
    score = float(get_model().predict_proba(build_features(req))[0, 1])
    decision = decision_for(score)
    return ScoreResponse(
        risk_score=round(score, 4), decision=decision,
        explanation="Synthetic model demonstration; not SHAP or a real fraud assessment",
        model_version=MODEL_VERSION, transaction_id=req.transaction_id,
    )


# Educational webinar 9 endpoint aliases; canonical routes remain /v1/*.
@app.get("/healthcheck")
def webinar_healthcheck() -> dict:
    return health()


@app.get("/model-info")
def webinar_model_info() -> dict:
    return model_info()


@app.post("/predict", response_model=ScoreResponse)
def webinar_predict(req: ScoreRequest) -> ScoreResponse:
    return score_transaction(req)
