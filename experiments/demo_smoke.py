"""Demonstrate a persisted CatBoost + SHAP through the FastAPI contract.

Uses only locally available educational CSV and model artifact.
Prints no user or transaction identifiers from the source dataset.
"""
import argparse
import json
import os

import pandas as pd
from fastapi.testclient import TestClient

from antifraud.app import app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--artifact", required=True)
    p.add_argument("--rows", type=int, default=3)
    args = p.parse_args()
    os.environ["ANTIFRAUD_MODEL_DIR"] = args.artifact
    os.environ["ANTIFRAUD_LLM_ENABLED"] = "false"
    df = pd.read_csv(args.data).sort_values(["timestamp", "transaction_id"]).tail(args.rows)
    with TestClient(app) as client:
        health = client.get("/health")
        health.raise_for_status()
        print("HEALTH", json.dumps(health.json(), ensure_ascii=False))
        for index, (_, row) in enumerate(df.iterrows(), start=1):
            payload = {
                "transaction_id": f"educational-demo-{index}",
                "user_id": "anonymous-demo",
                "amount": float(row["amount"]),
                "currency": "RUB",
                "timestamp": pd.to_datetime(row["timestamp"], utc=True).isoformat(),
                "country": str(row["country"]),
                "channel": str(row["channel"]),
                "hour": int(row["hour"]),
                "day_of_week": int(row["day_of_week"]),
                "avg_amount_30d": float(row["avg_amount_30d"]),
                "tx_count_24h": int(row["tx_count_24h"]),
                "tx_count_7d": int(row["tx_count_7d"]),
                "distinct_merchants_7d": int(row["distinct_merchants_7d"]),
                "distinct_countries_30d": int(row["distinct_countries_30d"]),
                "device_age_days": int(row["device_age_days"]),
                "is_new_device": int(row["is_new_device"]),
                "is_night_tx": int(row["is_night_tx"]),
            }
            response = client.post("/v1/score", json=payload)
            response.raise_for_status()
            result = response.json()
            print(json.dumps({
                "sample": index, "score": result["risk_score"],
                "decision": result["decision"],
                "explanation_method": result["explanation_method"],
                "explanation": result["explanation"],
                "top_factors": result["top_factors"],
            }, ensure_ascii=False))


if __name__ == "__main__":
    main()
