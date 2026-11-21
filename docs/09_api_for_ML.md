

## Актуальный статус проекта (21.11.2026)

Мы как команда используем **ExtraTrees v4** (200 деревьев, max_depth=4, min_samples_leaf=6; 13 исходных + 7 производных признаков) как финальную модель демонстрационного FastAPI-сервиса. Архив `extratrees-v4-artifact.zip` находится в корне GitHub, а `scripts/start_codespaces.py` приоритетно загружает его. Мы сравнили 32 конфигурации: validation PR-AUC **0,342852**, независимый test PR-AUC **0,237103** и test ROC-AUC **0,697491**. Отдельный [GitHub Actions workflow](https://github.com/SamDerden/Fraud_Detection_with-XAI_and_LLM/actions/runs/44890730210) завершился успешно: шесть проверок финальной v4, три демосценария и два API E2E-теста. Исторические RandomForest, CatBoost, LightGBM и Blend ниже описывают предыдущие этапы исследования, а не текущую модель.

**GroqCloud уже интегрирован в код** (`src/antifraud/llm.py`, `scripts/start_codespaces.py`): при доступном `GROQ_API_KEY` используется внешний провайдер для выбора подтверждённых диагностических факторов; без ключа или при ошибке действует локальный fallback. Реальный успешный сетевой ответ с пользовательским ключом **ещё не подтверждён**. Для v4 факторы — **score-delta, не SHAP**. Автономный агент с вызовами инструментов из вебинара 14 пока только спроектирован.

**Данные:** в дереве текущего публичного `main` нет исходного CSV v3 с 7 000 транзакций; там есть архив обученной модели и результаты экспериментов. Это не означает утраты исходного файла в других хранилищах. Для запуска сохранённой модели CSV не требуется, для переобучения и абляции признаков — требуется. Эксперимент удаления признаков устройства подготовлен, но его численные результаты пока не получены.

---

# Вебинар 9 — FastAPI для ML-сервиса

**Кейс:** антифрод на синтетических транзакциях. По лекции В09 путь запроса: клиент → API → Pydantic → модель → ответ; выбран REST/JSON, поскольку контракт прост и поддерживается большинством клиентов.

## Реализованные маршруты
| Маршрут | Метод | Результат |
|---|---|---|
| `/predict` (учебный алиас) | POST | risk score, decision, explanation |
| `/healthcheck` (учебный алиас) | GET | состояние |
| `/model-info` (учебный алиас) | GET | метаданные модели |
| `/v1/score` (основной) | POST | тот же скоринг по контракту |
| `/health`, `/v1/model-info` | GET | основные служебные маршруты |
| `/docs` | GET | Swagger/OpenAPI |

Полный JSON-контракт, обязательные признаки, типы и примеры ошибок находятся в [В03](03_contract.md). Pydantic возвращает 422 для отрицательной суммы, неверного времени и других невалидных полей. Основной сценарий Codespaces загружает сохранённую ExtraTrees v4 из ZIP. CatBoost и fallback — исторические/резервные режимы, не финальная модель и не production-банк.

## Проверка
```bash
PYTHONPATH=src uvicorn antifraud.app:app --host 127.0.0.1 --port 8000
curl -i http://127.0.0.1:8000/healthcheck
curl -i http://127.0.0.1:8000/model-info
# POST /predict: пример тела запроса — docs/03_contract.md
PYTHONPATH=src pytest -q tests/test_api.py tests/test_webinar_contract.py
```
**Доказательства:** в репозитории есть FastAPI, маршруты, TestClient-тесты и Swagger. Последний CI до этого обновления падал из-за устаревших тестовых данных, а не из-за отсутствия endpoint; тест исправлен с сохранением строгой проверки исторических счётчиков. Повторный CI требуется проверить.

## Ограничения
Нет подтверждённого HTTP E2E с итоговым ансамблем v3. Нет API-ключей или банковского платёжного шлюза. Переобучение при каждом запросе не выполняется для загруженного артефакта.

## Codespaces demo (update)
Added `.devcontainer/devcontainer.json` with Python 3.11 and forwarded port 8000. Run `PYTHONPATH=src uvicorn antifraud.app:app --host 0.0.0.0 --port 8000` then open `/docs`. Without v3 artifact, endpoint serves synthetic toy RF; SHAP E2E requires trusted `model.cbm` and `metadata.json` in Codespace with `ANTIFRAUD_MODEL_DIR`. This setup does not require local Docker.


## Актуальная модель для защиты: ExtraTrees v4

Выбрана **ExtraTrees v4 (200 деревьев, max_depth=4, min_samples_leaf=6)** с 13 исходными и 7 производными признаками: validation PR-AUC **0,342852**, ROC-AUC **0,803479**. Основание — лучший результат из [32 сравнительных конфигураций](WEBINAR05_32_EXPERIMENTS.md), с оговоркой о неодинаковом бюджете Optuna и отсутствии новой независимой test-оценки. [ZIP артефакта](../extratrees-v4-artifact.zip) уже в репозитории. **Финальная ExtraTrees v4 подключена к FastAPI; E2E API и отдельные тесты v4 успешно выполнены в GitHub Actions.** Старые метрики в документе сохранены как исторические, не как показатели выбранной модели.
