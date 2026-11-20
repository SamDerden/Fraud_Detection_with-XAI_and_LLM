# Вебинары 11 и 12 — контейнеризация и наблюдаемость

## В11 — Dockerfile и Compose
Методичка В11 **не требует фактического запуска Docker для сдачи**. Достаточно трёх корректных файлов и комментария к оптимизации. Наши файлы:
- [requirements.txt](../requirements.txt): зависимости FastAPI, Pydantic, scikit-learn, CatBoost, Prometheus.
- [Dockerfile](../Dockerfile): `python:3.11-slim` → WORKDIR → COPY requirements → pip install → COPY src → ENV PYTHONPATH → EXPOSE 8000 → CMD `uvicorn antifraud.app:app`.
- [docker-compose.yml](../docker-compose.yml): **два сервиса** `api` и `prometheus` с портами 8000/9090 и `depends_on`. Prometheus — разрешённый второй логичный компонент вместо PostgreSQL, потому что база логов в архитектуре не нужна.

**Чек-лист оптимизации:** slim-образ ✓; установка requirements до COPY src для Docker cache ✓; нет COPY всего репозитория ✓; pip --no-cache-dir ✓. **Что улучшить:** добавить/проверить `.dockerignore`, закрепить версии зависимостей и запуск от непривилегированного пользователя. Наличие конфигурации не равно успешному Docker build.

```bash
docker compose config -q
docker compose up --build
curl -i http://127.0.0.1:8000/healthcheck
```

## В12 — structured logging, метрики и дашборд
Методичка В12 требует middleware с JSON-логом (method, path, status, duration), `/metrics`, запросы включая 422, эскиз дашборда из **трёх блоков**, 3–5 сценариев алертов. В коде `src/antifraud/observability.py` middleware логирует JSON с **шаблоном маршрута**, методом, кодом ответа и временем. Не логируются транзакции, идентификаторы клиентов или payload. Метрики предоставляются в **Prometheus text exposition format**, а не JSON из учебного ноутбука — для существующей архитектуры это обоснованная замена.

**Реализованные метрики:** `antifraud_http_requests_total{method,route,status}`, `antifraud_http_request_duration_seconds_bucket`. HTTP 422 увеличивает счётчик со статусом 422. Для вычисления error rate и p95 используются запросы PromQL. Прямой факт сбора метрик на новой версии требует smoke.

### Эскиз дашборда
| Блок | Метрика | Визуализация |
|---|---|---|
| Инфраструктура | RPS, HTTP 5xx rate, p95 latency | 3 временных графика за 1 ч |
| Продукт | Доля транзакций, отправленных на ручную проверку; распределение по каналам | Линия и stacked bar; **ещё не инструментированы** |
| Модель | Score distribution, PSI и recall/FPR по подтверждённым delayed labels | Гистограмма + тренды; **ещё не инструментированы** |

### Сценарии алертов (учебный проектный план)
| Метрика | Порог | Серьёзность | Действие |
|---|---|---|---|
| HTTP 5xx rate | >1% за 5 мин | Warning | Проверить ошибки, конфигурацию модели |
| HTTP 5xx rate | >5% за 10 мин | Critical | Остановить релиз, откатить после решения дежурного |
| p95 latency | >500 мс за 5 мин | Warning | Проверить модель и нагрузку |
| Review rate | >2× baseline за 30 мин | Warning | Проверить сдвиг признаков; метрика ещё не реализована |
| PSI ключевого признака | >0.25 при ежедневном расчёте | Warning | Проверить данные и качество, без автоматического переобучения |

В `monitoring/alerts.yml` реализованы **два** правила Prometheus: 5xx и p95. Остальные три сценария — **проектные**, а не работающие уведомления. Alertmanager для доставки уведомлений не подключён.

### Воспроизводимый smoke
```bash
PYTHONPATH=src pytest -q tests/test_api.py tests/test_webinar12_observability.py
docker compose up --build
curl -s http://127.0.0.1:8000/metrics
```

**Сдача:** по В11 — три файла или ссылка на репозиторий; по В12 — ссылка на выполненный Colab с JSON-логами, `/metrics`, эскизом и алертами. Факт отправки преподавателю не подтверждён.
