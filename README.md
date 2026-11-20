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
*   **ML-ядро:** CatBoost (классический ML, табличные данные).
*   **XAI / LLM-слой:** Grok API для генерации текстовых объяснений.
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
Это **учебный прототип**, не готовый банковский антифрод. В репозитории есть FastAPI, синтетический fallback, код для сохранённой CatBoost-модели, тесты, Docker Compose, Prometheus и эксперименты. Grok и SHAP **ещё не интегрированы** в работающий API; текст объяснения прямо указывает на это. Порог модели выбран по валидации, вероятность не откалибрована. Никакой реальной блокировки платежей нет.

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
