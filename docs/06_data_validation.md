# Вебинар 6 — качество данных и Data Validation

**Единственный объект этой практики:** старый исходный файл `antifraud_transactions_dataset(1).csv` (5 000 строк, 20 столбцов). **Новый v3 в этом вебинаре не используется и не изменяется.**

## Что требует методичка
Изучить намеренно проблемный набор: `isna`, `nunique`, `describe`, поиск дубликатов/аномалий, правила Great Expectations, сравнение ручной и автоматической проверки, вывод. Учебный пример методички — интернет-магазин (312 × 5); здесь сознательно применён тот же метод к нашему антифроду.

## Результат повторного pandas-аудита исходного файла (20.11.2026)
| Правило | Нарушений |
|---|---:|
| Пропуски | 0 |
| Полные дубликаты | 0 |
| Повторяющиеся `transaction_id` | 0 |
| Неположительная `amount` | 0 |
| Некорректный `timestamp` | 0 |
| Не-бинарная `is_fraud` | 0 |
| `tx_count_7d < tx_count_24h` | **897** |
| `distinct_merchants_7d > tx_count_7d` | **768** |
| Оба противоречия одновременно | **541** |
| Хотя бы одно из двух | **1 124 (22,48%)** |

783 из 5 000 транзакций имеют метку фрода (15,66%). Два нарушения истории нельзя складывать арифметически: часть строк пересекается. Датасет не имеет пропусков и дублей, но **не проходит семантическую валидацию исторических агрегатов**.

## Повторяемая проверка
```bash
python experiments/webinar06_validate_legacy.py --data '/path/to/antifraud_transactions_dataset(1).csv' --output legacy_validation.json
python -m pip install 'great-expectations>=1,<2'
python experiments/webinar06_validate_legacy.py --data '/path/to/antifraud_transactions_dataset(1).csv' --output legacy_validation_ge.json --ge
```
Скрипт выводит JSON-отчёт pandas и опционально выполняет доступные проверки Great Expectations (обязательность ID, положительность суммы, бинарность метки). **Cross-column инварианты проверяются отдельно pandas.**

**Great Expectations не запущен:** библиотека отсутствует, установка из PyPI недоступна в текущей среде. Нельзя писать, что GE успешно обнаружил 1 124 ошибки. Предыдущий [Colab](https://colab.research.google.com/drive/1ZNyNejr5614Q046JCaN99ww0YkvclQoY?usp=sharing) и его результаты по намеренно испорченной учебной таблице — **другая практика/выборка**; не переносить оттуда числа пропусков или дубликатов на этот CSV.

## Решение и вывод
Первый датасет сохраняется **без исправлений** как документированный пример ошибок генерации. Его не используем для обучения итоговой модели. На вебинаре 6 показываем, что ручной pandas-анализ нашёл проблемы, которые базовые проверки схемы пропустили. Для официальной сдачи остаётся выполнить Great Expectations и предоставить ссылку на Colab и результаты.

## Next execution environment
Run the existing `experiments/webinar06_validate_legacy.py --ge` in Codespaces with Great Expectations installed, **only on the original 5,000-row flawed CSV**. Never use v3 for webinar 6. Current sandbox could not install Great Expectations; no GE success is claimed.
