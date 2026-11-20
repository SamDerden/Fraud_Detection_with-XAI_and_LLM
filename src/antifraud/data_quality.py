"""Input dataset checks for the educational antifraud pipeline."""
from __future__ import annotations

import pandas as pd

REQUIRED = (
    "transaction_id", "user_id", "amount", "currency", "timestamp",
    "merchant_id", "device_id", "country", "channel", "hour", "day_of_week",
    "avg_amount_30d", "tx_count_24h", "tx_count_7d",
    "distinct_merchants_7d", "distinct_countries_30d",
    "device_age_days", "is_new_device", "is_night_tx", "is_fraud",
)


def validate_dataset(frame: pd.DataFrame) -> dict:
    errors = []
    missing = sorted(set(REQUIRED) - set(frame.columns))
    if missing:
        return {"valid": False, "rows": len(frame), "errors": ["missing_columns: " + ", ".join(missing)]}
    if frame.empty:
        errors.append("empty_dataset")
    if frame[list(REQUIRED)].isna().any().any():
        errors.append("null_required_values")
    if frame["transaction_id"].duplicated().any():
        errors.append("duplicate_transaction_id")
    if not frame["is_fraud"].isin([0, 1]).all():
        errors.append("invalid_fraud_labels")
    if not frame["currency"].eq("RUB").all():
        errors.append("unexpected_currency")
    if not pd.to_numeric(frame["amount"], errors="coerce").gt(0).all():
        errors.append("nonpositive_or_invalid_amount")
    if not frame["hour"].between(0, 23).all() or not frame["day_of_week"].between(0, 6).all():
        errors.append("invalid_hour_or_weekday")
    for col in ["tx_count_24h", "tx_count_7d", "distinct_merchants_7d",
                "distinct_countries_30d", "device_age_days", "avg_amount_30d"]:
        if not pd.to_numeric(frame[col], errors="coerce").ge(0).all():
            errors.append("invalid_nonnegative_" + col)
    if not frame["is_new_device"].isin([0, 1]).all() or not frame["is_night_tx"].isin([0, 1]).all():
        errors.append("invalid_binary_features")
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if timestamps.isna().any():
        errors.append("invalid_timestamp")
    # A derived-feature consistency check; deviations are reported, not silently repaired.
    if ((frame["hour"] != timestamps.dt.hour) & timestamps.notna()).any():
        errors.append("hour_timestamp_mismatch")
    if ((frame["day_of_week"] != timestamps.dt.dayofweek) & timestamps.notna()).any():
        errors.append("weekday_timestamp_mismatch")
    warnings = []
    count_mismatch = int((frame["tx_count_7d"] < frame["tx_count_24h"]).sum())
    if count_mismatch:
        warnings.append(f"tx_count_7d_below_tx_count_24h: {count_mismatch} rows; check feature provenance")
    return {"valid": not errors, "rows": len(frame), "fraud_count": int(frame["is_fraud"].eq(1).sum()),
            "errors": errors, "warnings": warnings}
