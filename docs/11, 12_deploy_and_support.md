# Вебинары 11–12 — Docker и мониторинг

## Запуск
```bash
docker compose up --build
```
- API: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health
- Prometheus: http://127.0.0.1:9090
- Metrics: http://127.0.0.1:8000/metrics

Compose contains two services (`api`, `prometheus`), healthcheck and local loopback port bindings. Prometheus collects request counters and latency histograms. Logs include HTTP method, route template, status and duration; **no transaction/user IDs**.

`ANTIFRAUD_MODEL_DIR` points to a trusted local artifact folder with `model.cbm` and `metadata.json`; mount it inside the container and pass its path as an environment variable. Without it, the app runs an explicitly labeled synthetic fallback, not the CSV-trained model.

## Monitoring and alerts (design)
- 5xx error rate > 1% for 5 minutes: investigate.
- p95 HTTP latency > 500 ms for 5 minutes: investigate.
- Increased review rate: check feature quality and distribution shift.
- Drift: run `experiments/validate_and_drift.py` periodically and review PSI/KS report. Do not retrain automatically.
- Fraud recall/FPR are delayed-label metrics: **cannot** be inferred from live request logs without confirmed labels.

These are alert rules **proposed**, not configured in Alertmanager. No live production SLA is claimed.
