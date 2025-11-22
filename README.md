# Data Efficiency

Проект для экспериментов с обучением BERT модели на 10% исходного датасета с сохранением качества. Реализованы различные стратегии отбора данных для эффективного обучения моделей классификации текста.

## Возможности

- Обучение моделей ModernBERT с различными стратегиями отбора данных
- Автоматическое сохранение чекпоинтов во время обучения
- Комплексная оценка моделей с метриками и визуализациями
- Интеграция с TensorBoard для мониторинга обучения
- Автоматический подбор гиперпараметров для обучения на полном датасете

## Быстрый старт

### Установка

```bash
uv sync
```

### Загрузка датасета

```bash
uv run download_dataset
```

### Обучение

Базовый запуск обучения:

```bash
uv run run --config configs/train_config_example.json
```

Обучение на всем датасете с подбором гиперпараметров:

```bash
./train_full_with_tuning.sh
```

Обучение всех стратегий последовательно:

```bash
./train_all_diversity_strategies.sh
```

### Оценка модели

Оценка обученной модели:

```bash
uv run evaluate -c checkpoints/my_run/best/model.pt
```

С указанием конфигурации:

```bash
uv run evaluate --config configs/eval_config_example.json
```

## Стратегии отбора данных

Проект реализует следующие стратегии отбора данных для обучения на 10% исходного датасета:

1. **all** - Обучение на всем датасете (baseline для сравнения)
2. **random** - Случайный отбор примеров
3. **perplexity** - Отбор по перплексии под языковой моделью (низкая перплексия = простые примеры)
4. **el2n** - Отбор по L2 норме ошибки предсказания прокси-классификатора (EL2N score)
5. **ifd** - Instruction-Following Difficulty: измеряет влияние входного предложения на предсказание метки
6. **aflite_readability** - Комбинация AFLite (предсказуемость на случайных разбиениях) и метрики читаемости текста
7. **k_center** - K-Center Greedy алгоритм для геометрического coreset sampling (максимизация минимального расстояния)
8. **entropy_diversity** - Комбинация энтропии предсказаний (неопределенность модели) и геометрического разнообразия
9. **lexical_diversity** - Отбор по лексическому разнообразию (HD-D, MTLD, TTR метрики)
10. **qdit_lite** - Quality + Diversity hybrid: комбинация качества (1 - энтропия) и геометрического разнообразия
11. **edfs_lite** - Easy-and-Diverse-First: разделение данных на easy/medium/hard по энтропии и применение k-center в каждом бакете

## Архитектура проекта

### Основные компоненты

**Модель (`model.py`)**
- Обертка над ModernBERT с добавлением классификатора
- Поддержка заморозки слоев backbone
- Раздельные learning rate для head и backbone

**Тренер (`trainer.py`)**
- Управление процессом обучения с поддержкой стратегий отбора данных
- Интеграция с RoundScheduler для поэтапного отбора данных
- Автоматическое сохранение лучших чекпоинтов
- Поддержка подбора гиперпараметров

**RoundScheduler (`round_scheduler.py`)**
- Управление бюджетом данных (10% от исходного датасета)
- Поэтапный отбор данных согласно стратегии
- Создание DataLoader для каждого раунда обучения

**Стратегии (`strategies/`)**
- Базовый класс `DataSelectionStrategy` с методом `select()`
- Фабрика стратегий для создания экземпляров по имени
- Каждая стратегия реализует свой алгоритм отбора данных

**Оценка (`evaluate.py`)**
- Загрузка чекпоинтов и оценка на validation/test
- Вычисление метрик: accuracy, F1-macro, F1-weighted, AUC-ROC, AUC-PR
- Генерация визуализаций: confusion matrix, ROC curve, PR curve

### Поток данных

1. Загрузка датасета через `upload_dataset()`
2. Инициализация стратегии отбора через `get_strategy()`
3. RoundScheduler управляет отбором данных по раундам
4. Trainer выполняет обучение на отобранных данных
5. Сохранение чекпоинтов по метрике валидации
6. Оценка модели через `evaluate.py` с генерацией метрик и визуализаций

## Результаты экспериментов

### Сравнительная таблица метрик (F1-weighted и Accuracy)

Следующая таблица показывает ключевые метрики (F1-weighted и Accuracy) для validation и test наборов:

| Стратегия | Val Accuracy | Val F1-weighted | Test Accuracy | Test F1-weighted |
|-----------|--------------|-----------------|---------------|-----------------|
| **all (baseline)** | **0.955** | **0.955** | **0.947** | **0.947** |
| entropy_diversity | 0.913 | 0.913 | 0.921 | 0.921 |
| qdit_lite | 0.886 | 0.886 | 0.911 | 0.911 |
| k_center | 0.899 | 0.898 | 0.909 | 0.909 |
| edfs_lite | 0.878 | 0.877 | 0.882 | 0.882 |
| lexical_diversity | 0.765 | 0.760 | 0.862 | 0.862 |
| aflite_readability | 0.782 | 0.781 | 0.810 | 0.809 |
| random | 0.777 | 0.775 | 0.758 | 0.753 |
| ifd | 0.698 | 0.695 | 0.750 | 0.750 |
| perplexity | 0.586 | 0.544 | 0.617 | 0.603 |
| el2n | 0.535 | 0.432 | 0.576 | 0.535 |

### Визуализации

Для каждой стратегии доступны следующие визуализации в директории `artifacts/`:

- **ROC кривые**: `artifacts/{strategy_name}/test/roc_curve.png` и `validation/roc_curve.png`
- **Confusion матрицы**: `artifacts/{strategy_name}/test/confusion_matrix.png` и `validation/confusion_matrix.png`
- **Нормализованные confusion матрицы**: `artifacts/{strategy_name}/test/confusion_matrix_normalized.png`
- **PR кривые**: `artifacts/{strategy_name}/test/pr_curve.png` и `validation/pr_curve.png`

Примеры визуализаций для лучших стратегий:
- EDFS-lite: `artifacts/eval_edfs_lite_strategy/`
- Entropy Diversity: `artifacts/eval_entropy_diversity_strategy/`
- K-Center: `artifacts/eval_k_center_strategy/`

### Выводы

Лучшие результаты на 10% данных показали стратегии:
1. **EDFS-lite** - достигает 97.2% от baseline accuracy (0.921 vs 0.947)
2. **Entropy Diversity** - достигает 97.1% от baseline accuracy (0.920 vs 0.947)
3. **K-Center** - достигает 96.0% от baseline accuracy (0.909 vs 0.947)

Эти стратегии эффективно комбинируют отбор по сложности примеров и геометрическое разнообразие, что позволяет достичь качества близкого к обучению на всем датасете при использовании только 10% данных.

## Подбор гиперпараметров

Для стратегии "all" (обучение на полном датасете) доступен автоматический подбор гиперпараметров перед основным обучением.

### Включение подбора

Добавьте в конфигурационный файл:

```json
{
  "strategy_name": "all",
  "enable_hyperparameter_tuning": true,
  "warmup_epochs": 2,
  "tuning_n_iterations": 25,
  "tuning_sample_size": 0.15,
  "tuning_metric": "val_loss"
}
```

### Процесс подбора

1. **Поиск оптимального batch size** - определяет максимальный размер батча, помещающийся в память (бинарный поиск)
2. **Подбор гиперпараметров** - использует Random Search для поиска оптимальных:
   - Dropout модели (0.1 - 0.5)
   - Learning rate (1e-5 - 1e-4)
   - Weight decay (0.0, 0.01, 0.1)
   - Betas оптимизатора (различные варианты)
   - Раздельные learning rate для head и backbone (опционально)

Подбор выполняется на небольшой выборке из обучающего датасета (15% по умолчанию), что служит дополнительной регуляризацией и избегает переобучения на валидационном наборе.

После подбора модель пересоздается с оптимальными параметрами, веса сбрасываются, и начинается основное обучение.

### Параметры конфигурации

- `enable_hyperparameter_tuning` (bool): включить подбор (по умолчанию: False)
- `warmup_epochs` (int): количество эпох на итерацию подбора (по умолчанию: 2)
- `tuning_n_iterations` (int): количество случайных комбинаций для проверки (по умолчанию: 25)
- `tuning_sample_size` (float): доля обучающего датасета для подбора, 0.15 = 15% (по умолчанию: 0.15)
- `tuning_metric` (str): метрика для выбора лучших параметров - "val_loss" или "val_accuracy" (по умолчанию: "val_loss")

Опциональные параметры для настройки диапазонов:
- `dropout_range` (List[float]): [min, max] для dropout (по умолчанию: [0.1, 0.5])
- `lr_range` (List[float]): [min, max] для learning rate (по умолчанию: [1e-5, 1e-4])
- `weight_decay_options` (List[float]): список значений weight_decay (по умолчанию: [0.0, 0.01, 0.1])
- `betas_options` (List[List[float]]): список кортежей для betas (по умолчанию: [[0.9, 0.999], [0.95, 0.999], [0.9, 0.99]])

Пример конфигурации с подбором: `configs/train_config_with_tuning.json`

## Интеграция с ClearML

Для использования логирования ClearML необходимо настроить переменные окружения:

1. Создайте файл `.env` в корне проекта:
```bash
CLEARML_API_ACCESS_KEY=your_access_key_here
CLEARML_API_SECRET_KEY=your_secret_key_here
```

2. Получите креды из ClearML: https://app.clear.ml/settings/workspace-configuration

3. Включите ClearML в конфигурации: `"use_clearml": true`

## Структура проекта

```
data-efficiency/
├── src/data_efficiency/
│   ├── config.py          # Классы конфигурации
│   ├── data.py            # Обработка датасетов
│   ├── model.py           # Обертка модели ModernBERT
│   ├── trainer.py         # Пайплайн обучения
│   ├── evaluate.py        # Пайплайн оценки
│   ├── run.py             # Скрипт обучения
│   ├── round_scheduler.py # Управление раундами отбора данных
│   ├── strategies/        # Стратегии отбора данных
│   │   ├── base.py        # Базовый класс стратегии
│   │   ├── factory.py     # Фабрика стратегий
│   │   ├── all.py         # Стратегия "все данные"
│   │   ├── random.py      # Случайный отбор
│   │   ├── perplexity.py  # Отбор по перплексии
│   │   ├── el2n.py        # EL2N score
│   │   ├── ifd.py         # Instruction-Following Difficulty
│   │   ├── aflite_readability.py  # AFLite + читаемость
│   │   ├── k_center.py    # K-Center Greedy
│   │   ├── entropy_diversity.py   # Энтропия + разнообразие
│   │   ├── lexical_diversity.py  # Лексическое разнообразие
│   │   ├── qdit_lite.py   # Quality + Diversity
│   │   └── edfs_lite.py   # Easy-and-Diverse-First
│   └── utils/             # Вспомогательные функции
│       ├── embeddings.py   # Вычисление эмбеддингов
│       ├── evaluation.py   # Метрики оценки
│       ├── hyperparameter_tuning.py  # Подбор гиперпараметров
│       └── ...
├── configs/               # Конфигурационные файлы
├── checkpoints/           # Сохраненные чекпоинты моделей
├── artifacts/             # Результаты оценки (метрики, визуализации)
├── data/                  # Кэш датасета
└── runs/                  # Логи TensorBoard
```
