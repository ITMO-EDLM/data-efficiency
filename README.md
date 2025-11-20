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
uv run download_dataset
```

### Training

```bash
python -m data_efficiency.run
```
Add here path to config as `--config=my_filepath` if want to use custom model training or data selection parameters.

### Evaluation

Evaluate a trained model:

```bash
uv run evaluate -c checkpoints/my_run/best/model.pt
```
Add here path to config as `--config=my_filepath` if want to use custom model evaluation parameters.

For more details on evaluation, see [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md).

## Hyperparameter Tuning

For the "all" strategy (training on the full dataset), automatic hyperparameter tuning is available before the main training.

### Enabling Tuning

Add to your configuration file:

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

### What Tuning Does

1. **Finds optimal batch size** - determines the maximum batch size that fits in memory (binary search)
2. **Tunes hyperparameters** - uses Random Search to find optimal:
   - Model dropout (0.1 - 0.5)
   - Learning rate (1e-5 - 1e-4)
   - Weight decay (0.0, 0.01, 0.1)
   - Optimizer betas (various options)

Tuning is performed on a small sample from the train dataset (15% by default), which serves as additional regularization and avoids overfitting on the validation set.

After tuning, the model is recreated with optimal parameters, weights are reset, and main training begins.

### Configuration Parameters

- `enable_hyperparameter_tuning` (bool): enable tuning (default: False)
- `warmup_epochs` (int): number of epochs per tuning iteration (default: 2)
- `tuning_n_iterations` (int): number of random combinations to try (default: 25)
- `tuning_sample_size` (float): fraction of train dataset for tuning, 0.15 = 15% (default: 0.15)
- `tuning_metric` (str): metric for selecting best parameters - "val_loss" or "val_accuracy" (default: "val_loss")

Optional parameters for customizing ranges:
- `dropout_range` (List[float]): [min, max] for dropout (default: [0.1, 0.5])
- `lr_range` (List[float]): [min, max] for learning rate (default: [1e-5, 1e-4])
- `weight_decay_options` (List[float]): list of weight_decay values (default: [0.0, 0.01, 0.1])
- `betas_options` (List[List[float]]): list of tuples for betas (default: [[0.9, 0.999], [0.95, 0.999], [0.9, 0.99]])

Example configuration with tuning: `configs/train_config_with_tuning.json`

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
