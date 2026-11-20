# Лекция 9. Проектирование API для ML-сервисов.

**Ссылка на рабочий ноутбук:** [Наш Google Colab с API для ML](https://colab.research.google.com/drive/1nNWF5w7p7WEfIH3Gr2ggurOJiiupWGXI?usp=sharing)

## Отчёт по тестированию
Приложение создано, модель обучена.


## Актуальный FastAPI
Реализованы `GET /health`, `GET /v1/model-info`, `POST /v1/score`, а также `GET /metrics`. Swagger UI — `/docs`. Контракт и пример JSON находятся в `docs/03_contract.md`. По умолчанию запускается явно помеченный синтетический fallback. При наличии `ANTIFRAUD_MODEL_DIR` API загружает заранее обученный CatBoost из `model.cbm` и `metadata.json`, не переобучая его при каждом старте. Этот путь требует проверки реальным интеграционным запуском; score не калиброван и автоматической блокировки нет.


## SHAP и Grok — текущая интеграция
В режиме сохранённой CatBoost `POST /v1/score` возвращает `top_factors` с SHAP-вкладами в log-odds и `explanation_method`. SHAP рассчитывается через CatBoost native `ShapValues`, сумма проверяется на совпадение с raw model margin. Если `ANTIFRAUD_LLM_ENABLED=true`, `XAI_API_KEY` и `GROK_MODEL` настроены, Grok получает только имена признаков, направления и SHAP-вклады; выбирает разрешённые имена, а окончательный текст строится шаблоном. При ошибках LLM — детерминированный fallback. **Живой вызов Grok без ключа не подтверждён**. Автоматической блокировки нет.
