"""Run validation and chronological drift report against a local CSV."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from antifraud.data_quality import validate_dataset
from antifraud.drift import drift_report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output", default="data_quality_and_drift.json")
    args = p.parse_args()
    df = pd.read_csv(args.data)
    quality = validate_dataset(df)
    result = {"quality": quality}
    if quality["valid"] and len(df) >= 100:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values(["timestamp", "transaction_id"])
        pivot = int(len(df) * .8)
        result["drift"] = drift_report(df.iloc[:pivot], df.iloc[pivot:])
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    if not quality["valid"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
