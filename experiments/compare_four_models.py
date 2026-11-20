"""Reproducible four-candidate benchmark. Teaching data only; not banking decisions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, confusion_matrix

FEATURES = ["amount", "hour", "day_of_week", "avg_amount_30d", "tx_count_24h",
            "tx_count_7d", "distinct_merchants_7d", "distinct_countries_30d",
            "device_age_days", "is_new_device", "is_night_tx", "channel", "country"]
CATS = ["channel", "country"]


def threshold_at_fpr(y, scores, max_fpr=0.10):
    negatives = np.asarray(scores)[np.asarray(y) == 0]
    candidates = np.unique(np.r_[scores, 1.0 + 1e-9])
    allowed = [float(t) for t in candidates if np.mean(negatives >= t) <= max_fpr + 1e-12]
    return min(allowed) if allowed else 1.0 + 1e-9


def metrics(y, scores, threshold):
    pred = np.asarray(scores) >= threshold
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return {
        "average_precision": round(float(average_precision_score(y, scores)), 5),
        "roc_auc": round(float(roc_auc_score(y, scores)), 5),
        "threshold": round(float(threshold), 6),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 5),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 5),
        "false_positive_rate": round(float(cm[0, 1] / cm[0].sum()), 5),
        "confusion_matrix": cm.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Local teaching CSV; do not commit customer data")
    parser.add_argument("--output", default="four_model_comparison.json")
    parser.add_argument("--mlflow-uri", default=None, help="Optional local MLflow tracking URI")
    args = parser.parse_args()
    raw = Path(args.data).read_bytes()
    df = pd.read_csv(args.data)
    assert df.transaction_id.is_unique and not df.isna().any().any()
    assert set(df.is_fraud.unique()) == {0, 1}
    df["timestamp"] = pd.to_datetime(df.timestamp, utc=True)
    df = df.sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)
    n = len(df)
    if "split" in df.columns:
        labels = df["split"].astype(str).to_numpy()
        if set(labels) != {"train", "validation", "test"}:
            raise ValueError("Expected train/validation/test split labels")
        train, val, test = (np.flatnonzero(labels == name)
                            for name in ("train", "validation", "test"))
        if not (train.max() < val.min() and val.max() < test.min()):
            raise ValueError("Splits must be chronologically ordered")
        split_description = "explicit chronological split"
    else:
        train, val, test = np.split(np.arange(n), [int(n * .6), int(n * .8)])
        split_description = "chronological 60/20/20"
    forbidden = {"is_fraud", "transaction_id", "user_id", "merchant_id", "device_id", "split"}
    if forbidden.intersection(FEATURES):
        raise ValueError("Target or identifier leaked into model features")
    if set(FEATURES) - set(df.columns):
        raise ValueError("Missing model features")
    x = df[FEATURES].copy()
    x[CATS] = x[CATS].astype(str)
    encoded = pd.get_dummies(x, columns=CATS, dtype=int)
    y = df.is_fraud.to_numpy()
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=1),
        "CatBoost": CatBoostClassifier(iterations=350, depth=6, learning_rate=.05, loss_function="Logloss",
                                       verbose=False, random_seed=42, thread_count=2, cat_features=CATS),
        "LightGBM": LGBMClassifier(n_estimators=250, learning_rate=.04, num_leaves=15,
                                   min_child_samples=30, verbosity=-1, random_state=42, n_jobs=2),
    }
    validation, testing = {}, {}
    for name, model in models.items():
        features = x if name == "CatBoost" else encoded
        model.fit(features.iloc[train], y[train])
        validation[name] = model.predict_proba(features.iloc[val])[:, 1]
        testing[name] = model.predict_proba(features.iloc[test])[:, 1]
    weight = max(np.linspace(0, 1, 11),
                 key=lambda w: average_precision_score(y[val],
                     w * validation["CatBoost"] + (1 - w) * validation["LightGBM"]))
    name = "Blend CatBoost+LightGBM"
    validation[name] = weight * validation["CatBoost"] + (1 - weight) * validation["LightGBM"]
    testing[name] = weight * testing["CatBoost"] + (1 - weight) * testing["LightGBM"]
    results = {}
    for name in validation:
        threshold = threshold_at_fpr(y[val], validation[name])
        results[name] = {
            "validation": metrics(y[val], validation[name], threshold),
            "test": metrics(y[test], testing[name], threshold),
        }
    report = {
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "split": split_description,
        "rows": {"train": len(train), "validation": len(val), "test": len(test)},
        "fraud_counts": {"train": int(y[train].sum()), "validation": int(y[val].sum()), "test": int(y[test].sum())},
        "max_validation_false_positive_rate": .10,
        "blend_catboost_weight": float(weight),
        "results": results,
        "caveats": ["Educational dataset only", "Test not used for threshold selection",
                    "Blend weight tuned on validation: exploratory", "No probability calibration",
                    "Feature provenance and leakage require review"],
    }
    if args.mlflow_uri:
        import mlflow
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment("antifraud-four-model-benchmark")
        for model_name, result in results.items():
            with mlflow.start_run(run_name=model_name):
                mlflow.log_params({"model": model_name, "split": split_description,
                                   "seed": 42, "max_validation_fpr": 0.10})
                mlflow.log_metrics({"test_" + k: v for k, v in result["test"].items()
                                    if isinstance(v, (int, float))})
                mlflow.log_metrics({"validation_" + k: v for k, v in result["validation"].items()
                                    if isinstance(v, (int, float))})
                mlflow.set_tag("dataset_sha256", report["dataset_sha256"])
                if model_name == "Blend CatBoost+LightGBM":
                    mlflow.log_param("catboost_weight", float(weight))
        # Save the complete comparison as an artifact in the final run.
        with mlflow.start_run(run_name="benchmark-summary"):
            mlflow.log_dict(report, "four_model_comparison.json")
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
