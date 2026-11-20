"""Non-destructive preparation of the educational CSV. No imputation or fabricated labels."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

CORE = ["amount", "hour", "day_of_week", "country", "channel", "is_night_tx"]


def prepare(path: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(path)
    required = {"transaction_id", "user_id", "timestamp", "is_fraud",
                "tx_count_24h", "tx_count_7d", "distinct_merchants_7d",
                "device_age_days", "is_new_device", *CORE}
    missing = sorted(required - set(df))
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    if df.transaction_id.duplicated().any() or df[list(required)].isna().any().any():
        raise ValueError("Duplicate IDs or missing required values; manual review needed")
    timestamps = pd.to_datetime(df.timestamp, utc=True, errors="raise")
    annotated = df.copy()
    checks = {
        "flag_7d_below_24h": df.tx_count_7d < df.tx_count_24h,
        "flag_merchants_above_7d_tx": df.distinct_merchants_7d > df.tx_count_7d,
        "flag_hour_mismatch": df.hour != timestamps.dt.hour,
        "flag_weekday_mismatch": df.day_of_week != timestamps.dt.dayofweek,
        "flag_new_device_mismatch": df.is_new_device != (df.device_age_days < 7).astype(int),
        "flag_night_mismatch": df.is_night_tx != (df.hour <= 5).astype(int),
    }
    for name, mask in checks.items():
        annotated[name] = mask.astype("int8")
    annotated["has_consistency_warning"] = annotated[list(checks)].any(axis=1).astype("int8")
    annotated.to_csv(output / "transactions_annotated.csv", index=False)
    annotated.loc[:, ["transaction_id", "user_id", "timestamp", *CORE, "is_fraud"]].to_csv(
        output / "transactions_core_features.csv", index=False
    )
    annotated.loc[annotated.has_consistency_warning.eq(1)].to_csv(
        output / "flagged_transactions.csv", index=False
    )
    report = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": len(df), "fraud_count": int(df.is_fraud.sum()),
        "flag_counts": {key: int(value.sum()) for key, value in checks.items()},
        "rows_with_any_warning": int(annotated.has_consistency_warning.sum()),
        "rows_dropped": 0, "labels_fabricated": 0,
        "warning": "Upstream feature provenance is unknown; the core dataset is for sensitivity analysis only",
    }
    (output / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--out", default=Path("preprocessing_output"), type=Path)
    args = p.parse_args()
    print(json.dumps(prepare(args.data, args.out), ensure_ascii=False, indent=2))
