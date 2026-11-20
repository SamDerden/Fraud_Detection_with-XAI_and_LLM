"""Reproducible drift diagnostics, not automatic retraining."""
from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp


def psi(reference, current, bins=10):
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    if len(ref) < 20 or len(cur) < 20 or not np.isfinite(ref).all() or not np.isfinite(cur).all():
        raise ValueError("Need >=20 finite observations in each sample")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0 if np.array_equal(np.unique(ref), np.unique(cur)) else float("nan")
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(ref, bins=edges)[0] / len(ref)
    b = np.histogram(cur, bins=edges)[0] / len(cur)
    a, b = np.maximum(a, 1e-6), np.maximum(b, 1e-6)
    return float(np.sum((b - a) * np.log(b / a)))


def drift_report(reference, current, columns=("amount", "tx_count_24h", "device_age_days", "avg_amount_30d")):
    results = {}
    for col in columns:
        if col not in reference or col not in current:
            raise ValueError("Missing drift feature: " + col)
        a, b = reference[col].to_numpy(), current[col].to_numpy()
        stat, pvalue = ks_2samp(a, b)
        score = psi(a, b)
        results[col] = {"ks_statistic": float(stat), "ks_pvalue": float(pvalue),
                        "psi": score, "psi_alert": bool(np.isfinite(score) and score > .2)}
    return {"features": results, "action": "review_only_no_automatic_retraining",
            "note": "KS/PSI detect distribution change, not concept drift or quality degradation"}
