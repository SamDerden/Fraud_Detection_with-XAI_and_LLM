import pandas as pd
import numpy as np

from antifraud.data_quality import validate_dataset
from antifraud.drift import drift_report, psi


def valid_data():
    return pd.DataFrame({
        "transaction_id": ["t1", "t2"], "user_id": ["u1", "u2"],
        "amount": [100., 200.], "currency": ["RUB", "RUB"],
        "timestamp": ["2025-01-01T03:00:00Z", "2025-01-02T04:00:00Z"],
        "merchant_id": ["m1", "m2"], "device_id": ["d1", "d2"],
        "country": ["RU", "RU"], "channel": ["pos", "web"],
        "hour": [3, 4], "day_of_week": [2, 3],
        "avg_amount_30d": [100., 100.], "tx_count_24h": [1, 2],
        "tx_count_7d": [3, 4], "distinct_merchants_7d": [1, 2],
        "distinct_countries_30d": [1, 1], "device_age_days": [1, 5],
        "is_new_device": [1, 1], "is_night_tx": [1, 1], "is_fraud": [0, 1],
    })


def test_clean_dataset():
    assert validate_dataset(valid_data())["valid"]


def test_duplicate_rejected():
    d = valid_data()
    d.loc[1, "transaction_id"] = "t1"
    assert "duplicate_transaction_id" in validate_dataset(d)["errors"]


def test_timestamp_feature_mismatch_rejected():
    d = valid_data()
    d.loc[1, "hour"] = 20
    assert "hour_timestamp_mismatch" in validate_dataset(d)["errors"]


def test_invalid_label_rejected():
    d = valid_data()
    d.loc[1, "is_fraud"] = 2
    assert not validate_dataset(d)["valid"]


def test_drift_detection_shift():
    rng = np.random.default_rng(42)
    a = pd.DataFrame({"amount": rng.normal(100, 5, 400)})
    b = pd.DataFrame({"amount": rng.normal(300, 5, 400)})
    assert drift_report(a, b, columns=("amount",))["features"]["amount"]["psi_alert"]


def test_psi_same_distribution_small():
    a = np.arange(100, dtype=float)
    assert psi(a, a) < .001
