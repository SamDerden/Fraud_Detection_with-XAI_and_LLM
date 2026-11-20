# Вебинар 5 — трекинг экспериментов MLflow

**Датасет:** принятый v3, 7 000 синтетических транзакций. **Разбиение:** train 5 000, validation 1 000, test 1 000. Исходный датасет из вебинара 6 здесь не используется.

## Что требует методичка
1. Установить MLflow; обучить модель без трекинга.
2. Провести минимум 3 запуска с разными параметрами; получить `run_id`.
3. Записать параметры и метрики, свести результаты в одну таблицу.
4. Сохранить модель или график как MLflow artifact.
5. Написать вывод и сохранить ссылку на собственный Colab и скриншот/текст таблицы.

## Что сделано
- Ранее создан учебный [Colab](https://colab.research.google.com/drive/1YWUrK-zaxKjiuWNl0Cnm98mYjnYR5xKy?usp=sharing) с экспериментами RandomForest (старый учебный пример).
- Подготовлен `experiments/webinar05_mlflow_v3.py`: три RandomForest-конфигурации (100/6, 200/8, 350/12), логирование параметров, validation PR-AUC/ROC-AUC, SHA256 датасета, joblib-артефакт и CSV-сводка.
- Ранее проведено отдельное локальное диагностическое сравнение RF/CatBoost/LightGBM/Blend на v3; это **не** подтверждённый MLflow run. См. [V3_BENCHMARK.md](V3_BENCHMARK.md).

## Диагностическое сравнение четырёх моделей (не подменяет три MLflow run)
| Модель | Validation PR-AUC | Test PR-AUC | Test recall |
|---|---:|---:|---:|
| RandomForest | 0.25328 | 0.22334 | 0.40678 |
| CatBoost | 0.24352 | 0.21451 | 0.38983 |
| LightGBM | 0.20174 | 0.25524 | 0.38983 |
| Blend | **0.25779** | 0.22161 | **0.42373** |

В этом раннем четырёхмодельном сравнении по validation PR-AUC лидировал Blend; нельзя выбирать LightGBM только по test. CatBoost в API — прототип, а не победитель.

## Как воспроизвести MLflow
```bash
python -m pip install mlflow joblib pandas scikit-learn
python experiments/webinar05_mlflow_v3.py --data /path/to/transactions_v3.csv --tracking-uri file:./mlruns
mlflow ui --backend-store-uri file:./mlruns
```

**Фактический статус:** в доступной среде MLflow не установлен, а установка заблокирована отсутствием доступа к PyPI. Поэтому ни один новый `run_id` **не выдуман**, запуск не отмечен как успешный. Скрипт синтаксически проверен, но его MLflow-путь требует исполнения в среде с зависимостью. Для официальной сдачи также нужна личная ссылка на Colab и таблица run ID.

## Расширение: 16 семейств и 32 конфигурации (20.11.2026)

Проведён отдельный скрининг **16 семейств моделей × два набора признаков**, включая ExtraTrees, CatBoost, LightGBM, XGBoost и MLP. Полная таблица **всех 32 экспериментов**, PR-AUC, ROC-AUC, параметры Optuna, схема временного разбиения и ограничения приведены в [расширенном журнале вебинара 5](WEBINAR05_32_EXPERIMENTS.md). [CSV с результатами](../experiments/benchmark_16_results.csv). Лидер скрининга — **ExtraTrees + engineered (validation PR-AUC 0,342852)**. **Это только первичный поиск (8 trials для ExtraTrees), не углублённая оптимизация и не подтверждённый MLflow run.** Существующий CatBoost в API пока не заменён.

## Вывод
Трекинг делает сравнение воспроизводимым; выбирать модель по test нельзя. После 32-конфигурационного скрининга **выбран ExtraTrees v4 (validation PR-AUC 0,342852)**, но до окончательного выбора нужно завершить логирование и учесть эксплуатационные требования.

## Next execution environment
Use Codespaces Python environment (no local Docker), install `mlflow` and `joblib`, then run the existing `experiments/webinar05_mlflow_v3.py` **with the accepted v3 CSV**. The present sandbox still cannot install MLflow from PyPI; no run IDs are claimed.


## Решение по модели v4

Итоговый кандидат — **ExtraTreesClassifier, 200 деревьев, глубина 4, минимум 6 объектов в листе**, 13 исходных + 7 engineered признаков. На validation **PR-AUC 0,342852; ROC-AUC 0,803479**. Полный журнал [32 конфигураций](WEBINAR05_32_EXPERIMENTS.md) и [CSV](../experiments/benchmark_16_results.csv), дополнительный поиск [30+30 попыток](../experiments/deep_extratrees_results.json). Последний поиск ограничивал деревья 50–100 и дал 0,312462 для engineered, поэтому не опровергает конфигурацию с 200 деревьями. [Артефакт v4](../extratrees-v4-artifact.zip) опубликован, но ещё не интегрирован в API. **Это не MLflow run; run_id не придумываем.**
