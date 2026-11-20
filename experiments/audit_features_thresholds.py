"""Inspect dataset provenance risks and review thresholds without tuning on test."""
import argparse
import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, confusion_matrix

from antifraud.model_artifact import MODEL_FEATURES


def validation_threshold(labels, scores, max_fpr):
    negatives = np.asarray(scores)[np.asarray(labels) == 0]
    candidates = np.unique(np.r_[scores, 1.0 + 1e-9])
    return float(next((v for v in candidates if np.mean(negatives >= v) <= max_fpr), 1.0 + 1e-9))


def audit(data_path, model_path):
    data = pd.read_csv(data_path)
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data = data.sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)
    a, b = int(len(data) * .6), int(len(data) * .8)
    model = CatBoostClassifier()
    model.load_model(model_path)
    validation = model.predict_proba(data[list(MODEL_FEATURES)].iloc[a:b])[:, 1]
    test = model.predict_proba(data[list(MODEL_FEATURES)].iloc[b:])[:, 1]
    labels = data["is_fraud"].to_numpy()
    rows = []
    for cap in (.01, .02, .03, .05, .10, .15, .20):
        threshold = validation_threshold(labels[a:b], validation, cap)
        cm = confusion_matrix(labels[b:], test >= threshold, labels=[0, 1])
        rows.append({
            "validation_fpr_cap": cap,
            "threshold": round(threshold, 6),
            "test_recall": round(float(cm[1, 1] / cm[1].sum()), 5),
            "test_fpr": round(float(cm[0, 1] / cm[0].sum()), 5),
            "test_reviewed": int(cm[0, 1] + cm[1, 1]),
        })
    subgroup = {}
    pred = test >= validation_threshold(labels[a:b], validation, .10)
    for field in ("channel", "country"):
        subgroup[field] = {}
        for category, indices in data.iloc[b:].groupby(field).indices.items():
            idx = np.asarray(indices)
            negatives = labels[b:][idx] == 0
            if negatives.sum() < 20:
                continue
            subgroup[field][str(category)] = {
                "sample_size": len(idx),
                "negative_count": int(negatives.sum()),
                "fpr": round(float(pred[idx][negatives].mean()), 5),
            }
    return {
        "dataset_rows": len(data),
        "unique_users": int(data.user_id.nunique()),
        "users_shared_between_train_and_test": len(
            set(data.user_id.iloc[:a]) & set(data.user_id.iloc[b:])
        ),
        "tx_7d_less_than_tx_24h_count": int((data.tx_count_7d < data.tx_count_24h).sum()),
        "timestamp_hour_mismatch_count": int((data.timestamp.dt.hour != data.hour).sum()),
        "timestamp_weekday_mismatch_count": int((data.timestamp.dt.dayofweek != data.day_of_week).sum()),
        "test_average_precision": round(float(average_precision_score(labels[b:], test)), 5),
        "thresholds": rows,
        "subgroup_false_positive_rates": subgroup,
        "limitations": [
            "Rolling feature provenance cannot be verified from a flat CSV",
            "Repeated users across time splits may introduce correlated samples",
            "Seven-day transaction counts below 24-hour counts indicate synthetic inconsistency",
            "The threshold is selected on validation, not test",
            "Subgroup comparisons are descriptive, not fairness certification",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", default="feature_and_threshold_audit.json")
    args = parser.parse_args()
    result = audit(args.data, args.model)
    with open(args.out, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
