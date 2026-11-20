# Вебинар 5 — расширенный журнал экспериментов v3 (16 семейств, 32 конфигурации)

**Дата:** 20.11.2026. **Статус:** локальный скрининг, не зарегистрированные MLflow runs. **Исходные данные:** 7 000 синтетических транзакций; train 5 000, validation 1 000, test 1 000. Test для этого скрининга не использован.

## Методика

Сравнивались 16 семейств × два набора признаков: 13 исходных и 20 после feature engineering. Семь новых признаков: amount_to_avg, log_amount, log_avg_amount, tx24_share, merchants_per_tx, tx_per_day, log_device_age. Категориальные поля channel/country закодированы one-hot; числовая матрица стандартизована. Optuna TPE с seed=42: первые 80% train использовались для обучения в подборе, последние 20% train — для внутреннего временного holdout по PR-AUC; затем модель с лучшими внутренними параметрами переобучалась на всех 5 000 train и оценивалась на validation. **Бюджет неодинаков: 3–8 trials, поэтому ранжирование является разведочным.** Время выполнения и полный набор точных параметров доступны в исходном локальном CSV; в опубликованном CSV значения параметров округлены для читаемости.

## Полная таблица всех 32 конфигураций

| Модель | Признаки | Inner PR-AUC | Validation PR-AUC | ROC-AUC | Optuna trials |
|---|---|---:|---:|---:|---:|
| LogisticRegression | base | 0.238840 | 0.307751 | 0.801826 | 8 |
| LogisticRegression | engineered | 0.230745 | 0.287587 | 0.820850 | 8 |
| SGDClassifier | base | 0.249861 | 0.319495 | 0.692083 | 8 |
| SGDClassifier | engineered | 0.213153 | 0.293170 | 0.809354 | 8 |
| LDA | base | 0.239971 | 0.313104 | 0.812737 | 8 |
| LDA | engineered | 0.245406 | 0.316070 | 0.821231 | 8 |
| QDA | base | 0.259547 | 0.288769 | 0.802767 | 8 |
| QDA | engineered | 0.249200 | 0.269308 | 0.806468 | 8 |
| GaussianNB | base | 0.220001 | 0.220715 | 0.767568 | 8 |
| GaussianNB | engineered | 0.190808 | 0.269814 | 0.777868 | 8 |
| KNN | base | 0.256179 | 0.283832 | 0.710242 | 8 |
| KNN | engineered | 0.248837 | 0.236934 | 0.745454 | 8 |
| SVC | base | 0.263354 | 0.203162 | 0.573056 | 8 |
| SVC | engineered | 0.239919 | 0.221106 | 0.569419 | 8 |
| RandomForest | base | 0.268352 | 0.309827 | 0.811974 | 8 |
| RandomForest | engineered | 0.215252 | 0.269781 | 0.780055 | 8 |
| ExtraTrees | base | 0.275976 | 0.293906 | 0.773672 | 8 |
| ExtraTrees | engineered | 0.247572 | 0.342852 | 0.803479 | 8 |
| GradientBoosting | base | 0.205116 | 0.253103 | 0.803492 | 5 |
| GradientBoosting | engineered | 0.213853 | 0.236560 | 0.776113 | 5 |
| HistGradientBoosting | base | 0.200837 | 0.211276 | 0.753783 | 5 |
| HistGradientBoosting | engineered | 0.226469 | 0.241978 | 0.773189 | 5 |
| AdaBoost | base | 0.213717 | 0.288658 | 0.797553 | 5 |
| AdaBoost | engineered | 0.215235 | 0.278378 | 0.787660 | 5 |
| XGBoost | base | 0.211030 | 0.287735 | 0.808668 | 5 |
| XGBoost | engineered | 0.243287 | 0.255579 | 0.756759 | 5 |
| LightGBM | base | 0.203119 | 0.224752 | 0.750121 | 5 |
| LightGBM | engineered | 0.211976 | 0.269716 | 0.751621 | 5 |
| CatBoost | base | 0.235785 | 0.271098 | 0.724332 | 3 |
| CatBoost | engineered | 0.209338 | 0.293471 | 0.804268 | 3 |
| MLP | base | 0.120075 | 0.127574 | 0.502251 | 3 |
| MLP | engineered | 0.124877 | 0.263256 | 0.694143 | 3 |

## Выводы

Лидер этого скрининга: **ExtraTrees + engineered, validation PR-AUC 0.342852**, ROC-AUC 0.803479, inner holdout PR-AUC 0.247572; параметры: n_estimators=200, max_depth=4, min_samples_leaf=6. Это **не** углублённая оптимизация: всего 8 trials, один внутренний holdout, без оценки устойчивости по временным окнам и без повторного независимого test. Feature engineering повысил PR-AUC ExtraTrees с 0.293906 до 0.342852, но ухудшил часть других моделей.

Пять лучших семейств по validation: ExtraTrees 0.342852; SGDClassifier 0.319495; LDA 0.316070; RandomForest 0.309827; LogisticRegression 0.307751. Выбор максимума среди 32 validation-оценок создаёт риск переобучения на validation. Нельзя использовать прежний test для дальнейшего подбора: его метрики уже просматривались в предыдущих итерациях.

## Артефакты и воспроизведение

- [Таблица CSV: 32 конфигурации](https://github.com/SamDerden/Fraud_Detection_with-XAI_and_LLM/blob/main/experiments/benchmark_16_results.csv)
- [Скрипт воспроизведения и дальнейшей оптимизации ExtraTrees](https://github.com/SamDerden/Fraud_Detection_with-XAI_and_LLM/blob/main/experiments/deep_extratrees.py)
- [Исходный сравнительный отчёт v3](https://github.com/SamDerden/Fraud_Detection_with-XAI_and_LLM/blob/main/docs/V3_BENCHMARK.md)
- [Основной отчёт вебинара 5](https://github.com/SamDerden/Fraud_Detection_with-XAI_and_LLM/blob/main/docs/05_experiments_mlflow.md)

**Не заявляем:** завершённый MLflow-трекинг с run_id, глубокую Optuna-оптимизацию ExtraTrees, успешный тест финальной модели в API или превосходство на реальных банковских данных. Для окончательного выбора необходимы углублённый поиск и временная проверка; все новые эксперименты должны дописываться в журнал, а не подменять существующие 32 строки.


## Продолжение — углублённая временная проверка ExtraTrees (20.11.2026)

Проведён **реальный дополнительный прогон Optuna: 30 trials × 2 набора признаков = 60 попыток**, с тремя expanding-window разбиениями внутри train (0–2500 → 2500–3250; 0–3250 → 3250–4000; 0–4000 → 4000–5000). Для ускорения диапазон n_estimators ограничен **50–100**; поэтому эксперимент **не является точным повторением** предыдущего кандидата с 200 деревьями и не доказывает его несостоятельность. Test не использовался.

| Признаки | Optuna trials | Средний внутренний PR-AUC | Validation PR-AUC | Параметры победителя |
|---|---:|---:|---:|---|
| Base | 30 | 0.260458 | 0.287460 | n_estimators=100; max_depth=12; min_samples_leaf=12; min_samples_split=12; max_features=0.841897; bootstrap=true; class_weight=null |
| Engineered | 30 | 0.255381 | **0.312462** | n_estimators=50; max_depth=4; min_samples_leaf=30; min_samples_split=9; max_features=0.770864; bootstrap=false; class_weight=null |

Для base три внутренних значения: 0.253001, 0.284855, 0.243519. Для engineered: 0.260425, 0.280276, 0.225441. Лучший кандидат внутреннего CV не совпадает с лучшим на внешней validation: base 0.260458 против engineered 0.255381. Это показывает, что выбор по validation может менять победителя; **необходимо заранее фиксировать критерий и не подгонять под validation**. Предыдущий максимум 0.342852 не повторился в ограниченном поиске. Различия нельзя трактовать как доказательство превосходства одной конфигурации на новых данных.

[Код фактически выполненного прогона](../experiments/deep_extratrees_quick.py) · [Точные результаты JSON](../experiments/deep_extratrees_results.json). Полный широкий диапазон 200–700 деревьев и 60 trials на каждый набор признаков пытались запустить, но оба запуска завершились по тайм-ауту до публикации итогов. Их не выдаём за выполненные.
