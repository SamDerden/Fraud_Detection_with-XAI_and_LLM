# Вебинар 13 — проверяемая стратегия дрейфа

## Реализовано
`src/antifraud/drift.py` computes KS statistic, p-value and PSI on numerical features.
`experiments/validate_and_drift.py` compares chronological first 80% (reference) and last 20% (current) of a local CSV.

```bash
PYTHONPATH=src python experiments/validate_and_drift.py --data /path/to/antifraud_transactions_dataset.csv
```

Default features: `amount`, `tx_count_24h`, `device_age_days`, `avg_amount_30d`. PSI > 0.2 is a **review signal**, not automatic evidence of model degradation. KS p-values depend on sample size; multiple comparisons should be considered.

## Strategy
1. Track feature distributions, missing values, schema and traffic in a fixed reference window.
2. Investigate shift and provenance; separate genuine population shift from ETL defects.
3. Measure delayed-label recall, precision and false-positive rate once verified labels arrive.
4. Retrain only after approved data/label audit, experiment tracking and validation on future holdout.
5. Compare challenger vs champion in shadow mode, require human approval before rollout.

**Concept drift cannot be proven by feature-only PSI or KS.** This implementation is a reproducible drift diagnostic, not a continuous monitoring service or Evidently report. Evidently can be added once version compatibility and monitoring requirements are agreed.
