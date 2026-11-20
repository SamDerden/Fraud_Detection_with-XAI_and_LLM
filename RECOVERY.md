# Recovered runnable synthetic demonstration

The original five Colab notebooks remain unchanged on main. This branch adds a
minimal, installable FastAPI application inspired by the API contract in
`Семинар_9.ipynb` and `Семинар_11_12.ipynb`.

## Run

```bash
python -m pip install -r requirements.txt
PYTHONPATH=src uvicorn antifraud.app:app --reload
PYTHONPATH=src python -m pytest -q
```

Swagger: http://127.0.0.1:8000/docs

Example request:

```bash
curl -X POST http://127.0.0.1:8000/v1/score \
  -H 'Content-Type: application/json' \
  -d '{"transaction_id":"tx-1","user_id":"u-1","amount":12000,"timestamp":"2026-09-14T03:32:00Z"}'
```

## Critical limitations

- **Demo only:** the RandomForest is trained on synthetic features and toy labels
  generated from an explicit heuristic. Scores are **not calibrated real fraud
  probabilities**. Never use these outputs to block real transactions.
- No real transaction dataset, validated CatBoost model, SHAP explanation,
  Grok integration, drift monitoring, or production deployment is included.
- Thresholds 0.45/0.75 come from the Colab API prototype and are **not**
  empirically optimized.
- No ML model or dataset is downloaded from external URLs at startup.
- The original `Семинар_9.ipynb` contains a hard-coded ngrok auth token.
  **Revoke/rotate it immediately.** Deleting a line in a new commit does not
  remove it from Git history; rewrite history or remove the exposed file from
  public history after coordination with collaborators.
- GitHub CI results must be checked before merging; the branch creation does
  not establish a passing test run.

## Next step

Secure the exposed credential, identify an approved labeled dataset and its
schema, train/evaluate a single model with consistent preprocessing, persist
the model artifact, and only then add calibrated scores and explainability.
