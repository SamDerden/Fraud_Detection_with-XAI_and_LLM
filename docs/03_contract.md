# Вебинар 3 — контракт антифрод API (актуальная реализация)

## Назначение и версии
- HTTP API: `POST /v1/score`; версия API `v1` не равна версии модели.
- `GET /health` — состояние модели; `GET /v1/model-info` — схема и версия; `GET /docs` — Swagger UI.
- `GET /metrics` — метрики Prometheus, не предназначенные для публичного доступа в банковской сети.

## Запрос
```json
{
  "transaction_id": "txn_839201",
  "user_id": "usr_10482",
  "amount": 15700.50,
  "currency": "RUB",
  "timestamp": "2025-12-05T03:32:00Z",
  "country": "RU",
  "channel": "mobile",
  "hour": 3,
  "day_of_week": 4,
  "avg_amount_30d": 3200.0,
  "tx_count_24h": 10,
  "tx_count_7d": 22,
  "distinct_merchants_7d": 5,
  "distinct_countries_30d": 2,
  "device_age_days": 0,
  "is_new_device": 1,
  "is_night_tx": 1
}
```

In artifact mode (`ANTIFRAUD_MODEL_DIR` set), `avg_amount_30d`, `is_new_device`, `is_night_tx` are required; other engineered fields have defaults for backwards compatibility and **must be populated by a trustworthy feature provider in a realistic integration**. For the timestamp shown, day_of_week is 4 (Friday). The hour, weekday and device-derived indicators must be consistent. The API currently checks device/night consistency but not every historical aggregate; provenance must be validated upstream.

The synthetic fallback mode uses an older 12-feature toy model and is explicitly marked `demo_only=true`. Its scores and block decisions are **not banking decisions**.

## Response
```json
{
  "risk_score": 0.87,
  "decision": "review",
  "explanation": "Educational model score; explanation is not SHAP or LLM-generated",
  "model_version": "catboost-educational-v1",
  "transaction_id": "txn_839201",
  "demo_only": false
}
```

In artifact mode: `approve` or `review` only; **no automatic block**. Score is not a calibrated fraud probability. Threshold comes from validation data; test set is untouched. `demo_only=false` indicates a persisted CSV-trained model, **not production readiness**.

HTTP 422: invalid fields / missing engineered features; HTTP 500: unexpected server/model failure. Do not log transaction or user IDs. The model is not a payment authorization system.

## Security
Do not publish real bank data, identifiers, model secrets, API tokens or ngrok credentials. Restrict /metrics and Swagger when deployed outside a local demo.
