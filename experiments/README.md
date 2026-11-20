# Four-candidate antifraud benchmark

Uses a **local educational CSV**, not committed to this public repository.

```bash
python -m pip install -r experiments/requirements.txt
python experiments/compare_four_models.py --data /path/to/antifraud_transactions_dataset.csv --output four_model_comparison.json
```

Chronological 60/20/20 split; validation-only threshold selection at <=10% false-positive rate; untouched final test. Candidates: RandomForest, CatBoost, LightGBM and a validation-selected weighted CatBoost/LightGBM blend. The blend can select an endpoint (one model); it is not guaranteed to improve quality.

Do not interpret synthetic labels as real fraud probabilities. Audit feature provenance, possible leakage, calibration and fairness before any real use.


## Актуальная модель проекта (для защиты)

Выбрана **ExtraTrees v4**: 200 деревьев, глубина 4, минимум 6 объектов в листе; 13 исходных + 7 engineered признаков. Validation PR-AUC **0,342852**, ROC-AUC **0,803479** на синтетическом v3. Выбор по максимальному validation PR-AUC в [32-конфигурационном скрининге](../docs/WEBINAR05_32_EXPERIMENTS.md); дополнительный поиск (30+30 trials, 50–100 деревьев) не охватывал выбранные 200 деревьев. [ZIP модели](../extratrees-v4-artifact.zip) в `main`. **Загрузка v4 в FastAPI и XAI пока не реализованы/не подтверждены**. Legacy CatBoost/Blend в старых разделах — история экспериментов, не финальный выбор.
