# Data Efficiency

Код для экспериментов с обучением BERT модели на 10% исходного датасета с сохранением качества

## Features

- 🚀 Train ModernBERT models with data selection strategies
- 📊 Automatic checkpoint saving during training
- 🔍 Comprehensive model evaluation with metrics and visualizations
- 📈 TensorBoard integration for training monitoring

## Quick Start

### Installation

```bash
uv sync
```

### Download Dataset

```bash
download_dataset
```

### Training

```bash
python -m data_efficiency.run
```

### Evaluation

Evaluate a trained model:

```bash
evaluate -c checkpoints/my_run/best/model.pt
```

For more details on evaluation, see [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md).

## ClearML Integration

Для использования ClearML логирования необходимо настроить переменные окружения:

1. Создайте файл `.env` в корне проекта:
```bash
CLEARML_API_ACCESS_KEY=your_access_key_here
CLEARML_API_SECRET_KEY=your_secret_key_here
```

2. Получите креды из ClearML: https://app.clear.ml/settings/workspace-configuration

3. Включите ClearML в конфигурации: `"use_clearml": true`

## RunPod Setup

Для запуска обучения на RunPod см. подробную инструкцию: [RUNPOD_SETUP.md](RUNPOD_SETUP.md)

## Project Structure

```
data-efficiency/
├── src/data_efficiency/
│   ├── config.py          # Configuration classes
│   ├── data.py            # Dataset handling
│   ├── model.py           # ModernBERT model wrapper
│   ├── trainer.py         # Training pipeline
│   ├── evaluate.py        # Evaluation pipeline
│   ├── run.py             # Training script
│   ├── strategies/        # Data selection strategies
│   └── utils/             # Utility functions
├── checkpoints/           # Saved model checkpoints
├── artifacts/             # Evaluation results
├── data/                  # Dataset cache
└── runs/                  # TensorBoard logs
```