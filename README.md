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


### Первый подтверждённый локальный benchmark v3
Результаты четырёх кандидатов и ограничения: [docs/V3_BENCHMARK.md](docs/V3_BENCHMARK.md). MLflow и CI для этого запуска пока не подтверждены.


### Артефакт v3 и учебные маршруты вебинара 9
Локально экспортирована CatBoost-модель `catboost-educational-v3` с метаданными и проверкой SHAP. Архив передаётся отдельно от GitHub; проект не переключает API на v3 автоматически. Для использования распакуйте архив в `artifacts/catboost-v3` и установите `ANTIFRAUD_MODEL_DIR` на этот каталог. Добавлены совместимые учебные маршруты `POST /predict`, `GET /healthcheck`, `GET /model-info`; основными остаются `/v1/score`, `/health`, `/v1/model-info`. См. [отчёт](docs/V3_BENCHMARK.md). Проверки последнего коммита отслеживаются в GitHub Actions; live LLM и MLflow пока не подтверждены.


## Практика вебинаров 1–4: актуализирована
- [В01 — жизненный цикл](docs/01_lifecycle.md)
- [В02 — архитектура (Mermaid, 7 блоков)](docs/02_architecture.md)
- [В03 — текстовый контракт API](docs/03_contract.md)
- [В04 — MLOps-профиль и roadmap](docs/04_mlops_profile.md)
- [Контрольная ведомость В01–В04](docs/WEBINARS_01_04_REVIEW.md)

Документы отражают текущее состояние проекта: CatBoost остаётся техническим прототипом, финальная модель не утверждена; production, canary и MLflow v3 не объявляются выполненными. Официальная сдача в Holst/Google Docs требует отдельного подтверждения.


## Практика вебинаров 5–8 (20.11.2026)
- [В05 — MLflow на v3](docs/05_experiments_mlflow.md) и [скрипт трёх запусков](experiments/webinar05_mlflow_v3.py)
- [В06 — проверка **старого исходного датасета**](docs/06_data_validation.md) и [валидатор](experiments/webinar06_validate_legacy.py)
- [В07 — модельные тесты v3](docs/07_model_testing.md) и [pytest](tests/test_webinar07_quality.py)
- [В08 — unit/integration mock](docs/08_testing_strategy.md) и [pytest](tests/test_webinar08_components.py)
- [Контрольная ведомость В05–В08](docs/WEBINARS_05_08_REVIEW.md)

**Разделение данных:** В06 использует только исходный проблемный CSV 5 000 строк, сохраняемый без исправлений. В05 и В07 используют принятый v3 7 000 строк; В08 — mock. Локально тесты В07+В08: **17 passed** при предоставленном v3. MLflow/Great Expectations в текущей среде недоступны; их успешный запуск **не заявляется**. Не путать с официальной сдачей в Colab.


## Практика вебинаров 9–12 (20.11.2026)
- [В09 FastAPI](docs/09_api_for_ML.md)
- [В10 стратегия деплоя](docs/10_deploy_strategy.md)
- [В11 Docker и В12 мониторинг](docs/11%2C%2012_deploy_and_support.md)
- [Контрольная ведомость В09–В12](docs/WEBINARS_09_12_REVIEW.md)

Исправлена ошибка CI: mock-тест API теперь передаёт согласованные исторические счётчики; проверка в API сохранена. В12 middleware выдаёт JSON-логи, а `/metrics` — Prometheus text format. Фактический Docker build и окончательный E2E не объявляются подтверждёнными до успешного CI.


## Практика вебинаров 14–16
- [В14 — проект ИИ-агента](docs/14_AI_agent.md)
- [В15 — ИИ-ассистент и проверенный код](docs/15_prompt_engineering.md)
- [В16 — безопасность и AI code review](docs/16_vibe_coding.md)
- [Контрольная ведомость В14–В16](docs/WEBINARS_14_16_REVIEW.md)

В14 не требует реализации агента. В15 содержит код, тесты и **восстановленные**, а не дословно экспортированные промпты. В16 основан на лекции: отдельная практическая методичка не найдена. В13 остаётся отдельным этапом. Для будущей демонстрации выбран GitHub Codespaces без локального Docker.
