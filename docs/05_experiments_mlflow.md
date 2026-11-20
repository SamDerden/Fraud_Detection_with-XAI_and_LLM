# Лекция 5. Трекинг экспериментов (MLflowW&B)

**Ссылка на рабочий ноутбук:** [Наш Google Colab с кодом экспериментов](https://colab.research.google.com/drive/1YWUrK-zaxKjiuWNl0Cnm98mYjnYR5xKy?usp=sharing)

## Итоговый вывод
Мы провели 5 экспериментов, каждый раз меняя только один гиперпараметр RandomForest.

1. Самое сильное положительное влияние оказал class_weight="balanced":
   он значительно повысил Recall и F1-score относительно baseline.
   Это ожидаемо: при дисбалансе классов (~15% мошенничества) учёт весов критически важен.

2. Самое сильное отрицательное влияние оказали ограничения сложности модели:
   max_depth=6 и min_samples_leaf=10 заметно снизили F1 и особенно Recall.

3. Увеличение n_estimators с 100 до 300 почти не дало прироста качества.

ИТОГ: в задаче антифрода наиболее важным гиперпараметром оказался способ
учёта дисбаланса классов (class_weight), а не увеличение сложности/количества деревьев.


## Актуальная воспроизводимая реализация
`experiments/compare_four_models.py` сравнивает RandomForest, CatBoost, LightGBM и CatBoost/LightGBM blending на одном CSV, с хронологическим разбиением 60/20/20. Порог и вес смеси выбираются только по validation. При `--mlflow-uri file:./mlruns` скрипт записывает параметры, метрики и JSON-артефакт. Исходные пять учебных запусков RandomForest остаются историческим заданием и **не сравниваются напрямую** с новой выборкой.

```bash
python -m pip install -r experiments/requirements.txt
PYTHONPATH=src python experiments/compare_four_models.py --data /path/to/dataset.csv --mlflow-uri file:./mlruns
```

Скрипт и MLflow-интеграция добавлены в репозиторий; успешный запуск именно этой версии в CI/MLflow должен быть подтверждён отдельно.
