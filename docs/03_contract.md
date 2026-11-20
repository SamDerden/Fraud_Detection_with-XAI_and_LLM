# Вебинар 3 — текстовый контракт ML-сервиса

**Сервис:** учебный Antifraud Risk Scoring. **Основной endpoint:** `POST /v1/score`. **HTTP-метод:** POST. **Назначение:** принять признаки транзакции, вернуть оценку риска, решение для демонстрационного workflow и объяснение.

**Версия API:** `v1` в URL; **версия модели:** отдельное поле `model_version` в ответе. Вспомогательные маршруты: `GET /health`, `GET /v1/model-info`; учебные алиасы `POST /predict`, `GET /healthcheck`, `GET /model-info`.

## Входные поля (актуальный `ScoreRequest`)

| Поле | Тип | Обязательно | Пример / ограничение |
|---|---|---|---|
| `transaction_id` | string | да | `"demo-001"`, непустая |
| `user_id` | string | да | `"demo-user"`, непустая |
| `amount` | float | да | `1200.50`, строго положительное, конечное |
| `timestamp` | datetime | да | `"2026-02-15T00:20:32Z"`, с часовым поясом |
| `currency` | string | нет | `"RUB"`, только RUB |
| `country` | string | нет | `"RU"`, два символа |
| `channel` | enum | нет | `"mobile"`, также web/pos/api |
| `hour` | int | нет | `0`, диапазон 0–23 |
| `day_of_week` | int | нет | `6`, диапазон 0–6 |
| `avg_amount_30d` | float | условно | `1800.0`, >0; обязательно для артефакта |
| `tx_count_24h` | int | нет | `0`, >=0 |
| `tx_count_7d` | int | нет | `0`, >=0 и >= tx_count_24h при артефакте |
| `distinct_merchants_7d` | int | нет | `0`, >=0 и <= tx_count_7d при артефакте |
| `distinct_countries_30d` | int | нет | `0`, >=0 |
| `device_age_days` | float | нет | `41.48`, >=0 |
| `is_new_device` | int | условно | `0` или `1`, обязательно для артефакта |
| `is_night_tx` | int | условно | `1` или `0`, обязательно для артефакта |
| `avg_amount_ratio` | float | нет | `1.0`, >0; используется только fallback |

**Важно:** при `ANTIFRAUD_MODEL_DIR` модель требует согласованных производных признаков; `is_new_device` в текущем коде сравнивается с `int(device_age_days < 7)`, `is_night_tx` — с `int(hour <= 5)`. Это **ограничение действующего API**, которое ещё нужно согласовать с генератором v3 до финального E2E. `hour` и `day_of_week` сверяются с UTC timestamp. В режиме fallback требования отличаются. Не передавайте `is_fraud`: это обучающая метка, а не вход API.

## Полный пример запроса
```json
{
  "transaction_id": "demo-001", "user_id": "demo-user",
  "amount": 1200.5, "timestamp": "2026-02-15T00:20:32Z",
  "currency": "RUB", "country": "RU", "channel": "mobile",
  "hour": 0, "day_of_week": 6,
  "avg_amount_30d": 1800.0, "tx_count_24h": 0, "tx_count_7d": 0,
  "distinct_merchants_7d": 0, "distinct_countries_30d": 0,
  "device_age_days": 41.48, "is_new_device": 0, "is_night_tx": 1
}
```

## Выходные поля (`ScoreResponse`)
| Поле | Тип | Значение | Пример |
|---|---|---|---|
| `risk_score` | float | Оценка модели, **не** калиброванная вероятность реального фрода | `0.31` |
| `decision` | enum | approve/review/block; при CatBoost-артефакте только approve/review | `"approve"` |
| `explanation` | string | Пояснение решения | `"Демонстрационная оценка риска"` |
| `model_version` | string | Версия использованной модели | `"catboost-educational-v3"` (пример legacy; после интеграции v4 — `"extratrees-engineered-v4"`) |
| `transaction_id` | string | Идентификатор исходного запроса | `"demo-001"` |
| `demo_only` | bool | Признак демонстрационного режима в контракте | `false` |
| `explanation_method` | string/null | Способ формирования объяснения | `"shap"` |
| `top_factors` | array/null | Факторы объяснения | `[]` |

## Пример успешного ответа (иллюстративный, не результат live-запроса)
```json
{
  "risk_score": 0.31, "decision": "approve",
  "explanation": "Демонстрационная оценка риска",
  "model_version": "catboost-educational-v3",
  "transaction_id": "demo-001", "demo_only": false,
  "explanation_method": "shap", "top_factors": []
}
```
Значения score, explanation и top_factors **примерные**: настоящий ответ зависит от конкретного артефакта, SHAP и конфигурации LLM. `demo_only=false` означает режим загруженного CSV-артефакта в текущем коде, **не** банковскую готовность.

## Ошибки и поведение
| Ситуация | Код | Что видит клиент |
|---|---|---|
| Нет `amount`, отрицательная сумма, timestamp без timezone | **422** | Структурированная ошибка валидации Pydantic |
| Нет `avg_amount_30d` или бинарных производных признаков в режиме артефакта | **422** | `Missing required engineered features for artifact model` |
| `tx_count_7d < tx_count_24h` в режиме артефакта | **422** | `Inconsistent transaction history counts` |
| Файл модели отсутствует/повреждён | **500** в текущем коде (необработанная ошибка) | Нет гарантированного сообщения 503; **это технический долг** |

**Не утверждаем, что 503 реализован:** методичка приводит 503 как желательный код недоступности; текущий код не перехватывает исключения загрузки артефакта в 503.

## Правило совместимости
Изменение весов модели, порога или версии артефакта при неизменном JSON-контракте → меняем только `model_version`, оставляем `/v1/score`. Переименование обязательного поля или несовместимая смена типа → новый endpoint `/v2/score`. Добавление необязательного поля при обратной совместимости не требует v2.

## Что сдавать по В03
Этот документ содержит endpoint, метод, цель, таблицы полей, полный JSON-запрос и пример ответа, две и более ошибки, раздельные версии и правило обновления. Для официальной сдачи методичка просит ссылку на документ/область команды и примеры в чате курса; факт отправки не подтверждён.


## Актуализация выбора модели — ExtraTrees v4 (21.11.2026)

**Для защиты выбрана ExtraTrees v4**, 13 исходных + 7 производных признаков; validation PR-AUC **0,342852**, ROC-AUC **0,803479** на синтетическом v3. В 32-конфигурационном скрининге это лучший validation PR-AUC. Параметры: `n_estimators=200, max_depth=4, min_samples_leaf=6, random_state=42`. Артефакт [extratrees-v4-artifact.zip](../extratrees-v4-artifact.zip) опубликован. [Методика, все 32 результата и ограничения](WEBINAR05_32_EXPERIMENTS.md). **Сервер и XAI пока работают по legacy-контракту; v4 ещё не подключена.** Упоминания CatBoost/Blend выше относятся к предыдущему этапу проекта, не к финальному выбору. Отдельный test для v4 и калибровка порога не подтверждены.
