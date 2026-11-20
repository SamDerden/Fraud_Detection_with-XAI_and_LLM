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

Полный JSON-контракт, обязательные признаки, типы и примеры ошибок находятся в [В03](03_contract.md). Pydantic возвращает 422 для отрицательной суммы, неверного времени и других невалидных полей. При `ANTIFRAUD_MODEL_DIR` используется сохранённый CatBoost-прототип; без него — явно обозначенный синтетический fallback. **Это legacy-режим, не выбранная ExtraTrees v4 и не production-банк.**

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

Выбрана **ExtraTrees v4 (200 деревьев, max_depth=4, min_samples_leaf=6)** с 13 исходными и 7 производными признаками: validation PR-AUC **0,342852**, ROC-AUC **0,803479**. Основание — лучший результат из [32 сравнительных конфигураций](WEBINAR05_32_EXPERIMENTS.md), с оговоркой о неодинаковом бюджете Optuna и отсутствии новой независимой test-оценки. [ZIP артефакта](../extratrees-v4-artifact.zip) уже в репозитории. **В коде FastAPI и тестах пока используется legacy CatBoost/Blend или учебный fallback; E2E v4 ещё не подтверждён.** Старые метрики в документе сохранены как исторические, не как показатели выбранной модели.
