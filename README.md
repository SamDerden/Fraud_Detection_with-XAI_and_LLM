# Сервис детекции подозрительных транзакций в банке с объяснимостью (XAI) и LLM-слоем

## Описание проекта
Мы разрабатываем интеллектуальную систему для банка, которая в реальном времени определяет мошеннические транзакции. 

## Бизнес-цель
Снизить финансовые потери банка от мошенничества и уменьшить время ручной проверки алертов для операционистов за счет автоматической генерации пояснений.

## Команда и роли
*   **Участник 1 (Архитектор / AI Engineer):** Архитектура сервиса, API-контракты, деплой.
*   **Участник 2 (Data Scientist / ML Engineer):** Данные, обучение CatBoost, трекинг в MLflow.
*   **Участник 3 (QA / XAI Engineer):** Тестирование модели, интеграция с LLM, объяснимость.
*   **Участник 4 (MLOps / Tech Lead):** Мониторинг, валидация данных, документация, CI/CD.

## Технологический стек
*   **ML-ядро:** сравнение RandomForest, CatBoost, LightGBM и ансамбля; CatBoost пока используется в прототипе API, но не утверждён победителем для v3.
*   **XAI / LLM-слой:** SHAP и подключаемый LLM-слой; текущий адаптер Grok сохранён, провайдер может быть заменён.
*   **Сервис:** FastAPI, Docker.
*   **MLOps:** MLflow (трекинг экспериментов), Great Expectations (валидация данных), Evidently (мониторинг дрейфа).
*   **Тесты:** Pytest.

## Структура репозитория
```text
├── README.md                        # Этот файл
├── docs/                            # Отчеты по лекциям и итоговая документация
│   ├── 01_lifecycle.md              # Жизненный цикл ИИ-системы
│   ├── 02_architecture.md           # Архитектура сервиса
│   ├── 03_contract.md               # Контракт ML-сервиса
│   ├── 04_mlops_profile.md          # MLOps-профиль компании
│   ├── 05_experiments_mlflow.md     # MLOps эксперименты
│   ├── 06_data_validation.md        # Валидация данных
│   ├── 07_model_testing.md          # Валидация данных
│   ├── 08_testing_strategy.md       # Стратегия тестирования
│   ├── 09_api_for_ML.md             # Проектирование API для ML
│   ├── 10_deploy_strategy.md        # Стратегия деплоя
│   ├── 11, 12_deploy_and_support.md # Деплой и поддержка
│   ├── 13_drift_strategy.md         # Стратегия обнаружения дрейфа
│   ├── 14_AI_agent.md               # Проектирование ИИ агента
│   ├── 15_prompt_engineering.md     # Использование ИИ-ассистентов и промпт-инжиниринг
│   ├── 16_vibe_coding.md            # Ответственный ИИ, мониторинг и ограничения


## Реальный статус реализации (20.11.2026)
Это **учебный прототип**, не готовый банковский антифрод. В репозитории есть FastAPI, синтетический fallback, код для сохранённой CatBoost-модели, тесты, Docker Compose, Prometheus и эксперименты. SHAP интегрирован в API для сохранённой CatBoost; Grok подключается опционально через API-ключ, но живой вызов пока не подтверждён. Порог модели выбран по валидации, вероятность не откалибрована. Никакой реальной блокировки платежей нет.

### Локальные эксперименты
```bash
python -m pip install -r experiments/requirements.txt
PYTHONPATH=src python experiments/validate_and_drift.py --data /path/to/antifraud_transactions_dataset.csv
PYTHONPATH=src python experiments/compare_four_models.py --data /path/to/antifraud_transactions_dataset.csv --mlflow-uri file:./mlruns
PYTHONPATH=src python experiments/train_deployable.py --data /path/to/antifraud_transactions_dataset.csv --out artifacts/catboost-educational-v1 --mlflow-uri file:./mlruns
```

**Данные не коммитить:** CSV должен быть учебным/синтетическим и разрешённым к использованию. `mlruns/` и `artifacts/` исключены через `.gitignore`.

### API с сохранённой моделью
```bash
python -m pip install -r requirements.txt
ANTIFRAUD_MODEL_DIR=artifacts/catboost-educational-v1 PYTHONPATH=src uvicorn antifraud.app:app --host 127.0.0.1 --port 8000
```
Swagger: http://127.0.0.1:8000/docs. Запрос — см. `docs/03_contract.md`. Для Docker:
```bash
ANTIFRAUD_MODEL_DIR=/models/catboost-educational-v1 docker compose up --build
```
Compose монтирует локальную папку `./artifacts` внутрь `/models`. Без переменной API использует демонстрационную модель, обучаемую при первом запросе. Для безопасной демонстрации это допустимо; для реальной эксплуатации — нет.

### Проверки
```bash
PYTHONPATH=src pytest -q
docker compose config -q
```
GitHub Actions также проверяет Docker build. Наличие кода не заменяет успешный CI и ручной smoke-тест. **Ранее опубликованный ngrok-токен необходимо отозвать у провайдера; изменение ноутбука не очищает Git-историю.**


## SHAP + optional Grok (educational API)

When `ANTIFRAUD_MODEL_DIR` points to a trained CatBoost artifact, `POST /v1/score` returns:
- `risk_score` — uncalibrated model score;
- `decision` — `approve` or `review`, **never automatic block**;
- `top_factors` — top 5 CatBoost native SHAP contributions with direction and log-odds contribution;
- `explanation` — Russian explanation of the verified factors;
- `explanation_method` — `shap_template`, `grok_grounded_selection`, or a fallback code.

**SHAP values are log-odds contributions, not probability percentage points.** The code checks that SHAP contributions plus base value reconstruct the raw CatBoost prediction.

Grok is **optional and disabled by default**. It can select/order the supplied SHAP factor names, but cannot introduce new factors into the final text. The final text is deterministically rendered from verified SHAP data. Only feature names, directions and SHAP values are sent externally, never transaction/user identifiers or raw payment details. Missing credentials, model name, API errors, invalid JSON and unapproved factors fall back to the local template.

To enable a live Grok request, set `ANTIFRAUD_LLM_ENABLED=true`, `XAI_API_KEY` in a secret manager, and `GROK_MODEL` to an available model identifier. No credentials are stored in this repository. **Live Grok access is not proven by mock tests.**

### Dataset integrity and thresholds
```bash
PYTHONPATH=src python experiments/audit_features_thresholds.py \
  --data /path/to/antifraud_transactions_dataset.csv \
  --model artifacts/catboost-educational-v1/model.cbm
```
The report compares validation-selected review thresholds at 1%, 2%, 3%, 5%, 10%, 15%, 20% validation FPR caps, reports test performance, overlapping users and 7-day/24-hour count inconsistencies. **Do not optimize thresholds on the test set.** Historical feature provenance and true concept drift remain unverified.


### Демонстрация API + SHAP без Grok-ключа
```bash
PYTHONPATH=src python experiments/demo_smoke.py \
  --data /path/to/antifraud_transactions_dataset.csv \
  --artifact artifacts/catboost-educational-v1 --rows 3
```
The script sends three educational examples through FastAPI TestClient and prints score, review/approve decision, SHAP factor names, and grounded Russian explanation. It does not print original transaction/user IDs. It deliberately disables external LLM calls.


## Новый основной учебный датасет: v3 (7 000 транзакций)

**Решение проекта:** сохраняем синтетический v3, не смешиваем его с v1/v2 и не исправляем задним числом по результатам теста. История warm-up (3 776 событий) используется только для расчёта признаков, не входит в 7 000 размеченных операций. Данные v3: 220 пользователей, 321 положительная метка (4,59%). Явные хронологические выборки: **train 5 000 / validation 1 000 / test 1 000**. Данные хранятся вне GitHub до согласованной публикации, генератор и контроль качества должны быть сохранены вместе с ними для воспроизводимости.

**Аудит:** обнаружена сильная связь нового устройства с меткой фрода (41,30% среди новых устройств против 3,34% среди привычных). Это свойство симулятора, а не доказательство утечки или повод автоматически менять данные. Диагностическое исключение `is_new_device` и `device_age_days` снижало PR-AUC на тесте у всех четырёх вариантов; эти результаты сохраняем как анализ чувствительности, а не основание для выбора победителя. Обобщение на полностью новых клиентов ещё не проверено. Данные синтетические и не валидированы на реальном банковском потоке.

**Этапы:** (1) аудит данных — проведён; (2) воспроизводимое сравнение четырёх моделей на v3 — следующий этап; (3) выбор порога по validation и независимая оценка test; (4) SHAP; (5) LLM и API; (6) эксплуатационные проверки. Существующий CatBoost API обучен на старой версии данных и **не является финальной моделью v3**.

Запуск сравнительного эксперимента после размещения v3 локально:

```bash
python -m pip install -r experiments/requirements.txt
PYTHONPATH=src python experiments/compare_four_models.py --data /path/to/transactions_v3.csv --output v3_four_models.json --mlflow-uri file:./mlruns
```

Скрипт учитывает явный столбец `split` и не заменяет разбиение v3 на 60/20/20. Столбцы `is_fraud`, `split` и идентификаторы не входят в признаки модели. Test нельзя использовать для настройки гиперпараметров или порога. Подробности: [этап экспериментов](docs/05_experiments_mlflow.md) и [тестирование](docs/07_model_testing.md).


## Единый реестр практик (вебинары 1–16)
[Покрытие заданий, артефакты и честный статус проверки](docs/PRACTICE_COVERAGE.md). Этот документ фиксирует, что уже есть в репозитории, и что ещё нельзя считать подтверждённым. Скрипты `experiments/compare_four_models.py` и `experiments/train_deployable.py` учитывают явное разбиение v3 на 5 000/1 000/1 000.
