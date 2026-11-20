"""Train a reproducible *educational* CatBoost model from local validated data.

Uses 60% train, 20% validation (threshold), 20% untouched test.
The exported model is trained on the train portion only; metadata records the split.
Never use this artifact for real banking decisions.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix

from antifraud.data_quality import validate_dataset
from antifraud.model_artifact import MODEL_FEATURES, CATEGORICAL


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="artifacts/catboost-educational-v1")
    p.add_argument("--mlflow-uri", default=None, help="Optional local/file MLflow tracking URI")
    args = p.parse_args()
    raw = Path(args.data).read_bytes()
    df = pd.read_csv(args.data)
    check = validate_dataset(df)
    if not check["valid"]:
        raise ValueError("Dataset validation failed: " + ", ".join(check["errors"]))
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)
    if "split" in df.columns:
        labels = df["split"].astype(str).to_numpy()
        if set(labels) != {"train", "validation", "test"}:
            raise ValueError("Expected train/validation/test split labels")
        import numpy as np
        train_idx, val_idx, test_idx = (np.flatnonzero(labels == name)
                                        for name in ("train", "validation", "test"))
        if not (train_idx.max() < val_idx.min() and val_idx.max() < test_idx.min()):
            raise ValueError("Splits must be chronological")
        a, b = len(train_idx), len(train_idx) + len(val_idx)
        split_description = "explicit chronological split"
    else:
        a, b = int(len(df) * .6), int(len(df) * .8)
        split_description = "chronological 60/20/20"
    forbidden = {"is_fraud", "transaction_id", "user_id", "merchant_id", "device_id", "split"}
    if forbidden.intersection(MODEL_FEATURES):
        raise ValueError("Target or identifier leaked into model features")
    x = df[list(MODEL_FEATURES)].copy()
    for col in CATEGORICAL:
        x[col] = x[col].astype(str)
    y = df["is_fraud"]
    model = CatBoostClassifier(iterations=350, depth=6, learning_rate=.05,
                               loss_function="Logloss", verbose=False, random_seed=42,
                               thread_count=2, cat_features=list(CATEGORICAL))
    model.fit(x.iloc[:a], y.iloc[:a])
    val_scores = model.predict_proba(x.iloc[a:b])[:, 1]
    negatives = val_scores[y.iloc[a:b].to_numpy() == 0]
    candidates = sorted(set(float(v) for v in val_scores) | {1.0 + 1e-9})
    threshold = next((v for v in candidates if (negatives >= v).mean() <= .10), 1.0 + 1e-9)
    test_scores = model.predict_proba(x.iloc[b:])[:, 1]
    test_pred = test_scores >= threshold
    cm = confusion_matrix(y.iloc[b:], test_pred, labels=[0, 1])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out / "model.cbm"))
    meta = {
        "model_type": "CatBoostClassifier",
        "model_version": "catboost-educational-v1",
        "features": list(MODEL_FEATURES),
        "categorical": list(CATEGORICAL),
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "rows": {"train": a, "validation": b - a, "test": len(df) - b},
        "split": split_description,
        "review_threshold": min(.999999, float(threshold)),
        "block_threshold": None,
        "test_average_precision": float(average_precision_score(y.iloc[b:], test_scores)),
        "test_roc_auc": float(roc_auc_score(y.iloc[b:], test_scores)),
        "test_recall": float(cm[1, 1] / cm[1].sum()),
        "test_fpr": float(cm[0, 1] / cm[0].sum()),
        "test_confusion_matrix": cm.tolist(),
        "disclaimer": "Synthetic educational data; review only; no automatic blocking",
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if args.mlflow_uri:
        import mlflow
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment("antifraud-educational")
        with mlflow.start_run(run_name="catboost-deployable-v1"):
            mlflow.log_params({"model": "CatBoost", "iterations": 350, "depth": 6,
                               "split": split_description, "seed": 42})
            mlflow.log_metrics({k: meta[k] for k in ("test_average_precision", "test_roc_auc", "test_recall", "test_fpr")})
            mlflow.log_artifacts(str(out), artifact_path="model")
            mlflow.set_tag("dataset_sha256", meta["dataset_sha256"])
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
