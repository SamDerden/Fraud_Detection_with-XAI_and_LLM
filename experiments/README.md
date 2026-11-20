# Four-candidate antifraud benchmark

Uses a **local educational CSV**, not committed to this public repository.

```bash
python -m pip install -r experiments/requirements.txt
python experiments/compare_four_models.py --data /path/to/antifraud_transactions_dataset.csv --output four_model_comparison.json
```

Chronological 60/20/20 split; validation-only threshold selection at <=10% false-positive rate; untouched final test. Candidates: RandomForest, CatBoost, LightGBM and a validation-selected weighted CatBoost/LightGBM blend. The blend can select an endpoint (one model); it is not guaranteed to improve quality.

Do not interpret synthetic labels as real fraud probabilities. Audit feature provenance, possible leakage, calibration and fairness before any real use.
