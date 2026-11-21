> **Обновление XAI (21.11.2026).** Для финальной ExtraTrees v4 реализован TreeSHAP с проверкой аддитивности: базовое значение + сумма SHAP-вкладов = прогноз класса мошенничества. В интерфейсе связанные закодированные и производные признаки суммируются по смысловым группам. В GroqCloud передаются только проверенные названия и вклады SHAP, без персональных данных. Модель и её порог не переобучались. Новый CI-прогон требуется для подтверждения внедрения; старые результаты score-delta относятся к предыдущей версии интерфейса.



## Актуальный статус проекта (21.11.2026)

Мы как команда используем **ExtraTrees v4** (200 деревьев, max_depth=4, min_samples_leaf=6; 13 исходных + 7 производных признаков) как финальную модель демонстрационного FastAPI-сервиса. Архив `extratrees-v4-artifact.zip` находится в корне GitHub, а `scripts/start_codespaces.py` приоритетно загружает его. Мы сравнили 32 конфигурации: validation PR-AUC **0,342852**, независимый test PR-AUC **0,237103** и test ROC-AUC **0,697491**. Отдельный [GitHub Actions workflow](https://github.com/SamDerden/Fraud_Detection_with-XAI_and_LLM/actions/runs/44890730210) завершился успешно: шесть проверок финальной v4, три демосценария и два API E2E-теста. Исторические RandomForest, CatBoost, LightGBM и Blend ниже описывают предыдущие этапы исследования, а не текущую модель.

**GroqCloud уже интегрирован в код** (`src/antifraud/llm.py`, `scripts/start_codespaces.py`): при доступном `GROQ_API_KEY` используется внешний провайдер для выбора подтверждённых диагностических факторов; без ключа или при ошибке действует локальный fallback. Реальный успешный сетевой ответ с пользовательским ключом **ещё не подтверждён**. Для v4 факторы — **score-delta, не SHAP**. Автономный агент с вызовами инструментов из вебинара 14 пока только спроектирован.

**Данные:** в дереве текущего публичного `main` нет исходного CSV v3 с 7 000 транзакций; там есть архив обученной модели и результаты экспериментов. Это не означает утраты исходного файла в других хранилищах. Для запуска сохранённой модели CSV не требуется, для переобучения и абляции признаков — требуется. Эксперимент удаления признаков устройства подготовлен, но его численные результаты пока не получены.

---

# Эксперименты Antifraud AI: финальная ExtraTrees v4 и исторические кандидаты

Uses a **local educational CSV**, not committed to this public repository.

```bash
python -m pip install -r experiments/requirements.txt
python experiments/compare_four_models.py --data /path/to/antifraud_transactions_dataset.csv --output four_model_comparison.json
```

Chronological 60/20/20 split; validation-only threshold selection at <=10% false-positive rate; untouched final test. Candidates: RandomForest, CatBoost, LightGBM and a validation-selected weighted CatBoost/LightGBM blend. The blend can select an endpoint (one model); it is not guaranteed to improve quality.

Do not interpret synthetic labels as real fraud probabilities. Audit feature provenance, possible leakage, calibration and fairness before any real use.


## Актуальная модель проекта (для защиты)

Выбрана **ExtraTrees v4**: 200 деревьев, глубина 4, минимум 6 объектов в листе; 13 исходных + 7 engineered признаков. Validation PR-AUC **0,342852**, ROC-AUC **0,803479** на синтетическом v3. Выбор по максимальному validation PR-AUC в [32-конфигурационном скрининге](../docs/WEBINAR05_32_EXPERIMENTS.md); дополнительный поиск (30+30 trials, 50–100 деревьев) не охватывал выбранные 200 деревьев. [ZIP модели](../extratrees-v4-artifact.zip) в `main`. **ExtraTrees v4 подключена к FastAPI и диагностике score-delta; автоматические тесты v4 прошли в GitHub Actions**. Legacy CatBoost/Blend в старых разделах — история экспериментов, не финальный выбор.
