import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data_efficiency.data import TokenizedDataset
from data_efficiency.model import ModernBert
from data_efficiency.utils.data import build_dataloader
from data_efficiency.utils.loss import get_loss


def find_optimal_batch_size(
    model: ModernBert,
    dataset: TokenizedDataset,
    device: str,
    initial_batch_size: int = 64,
    max_iterations: int = 10,
) -> int:
    """
    Бинарный поиск максимального batch size без OOM.

    Args:
        model: Модель для тестирования
        dataset: Датасет для тестирования
        device: Устройство (cuda/mps/cpu)
        initial_batch_size: Начальный batch size
        max_iterations: Максимальное количество итераций поиска

    Returns:
        optimal_batch_size: Найденный оптимальный batch size
    """
    min_batch = 1
    max_batch = initial_batch_size * 4
    optimal_batch = initial_batch_size

    print(f"Searching for optimal batch size (initial: {initial_batch_size}, max: {max_batch})...")

    for iteration in range(max_iterations):
        if max_batch - min_batch < 2:
            break

        test_batch = (min_batch + max_batch) // 2
        print(f"  Testing batch size: {test_batch} (range: {min_batch}-{max_batch})")

        try:
            # Создаем небольшой dataloader для теста
            test_loader = build_dataloader(
                dataset,
                batch_size=test_batch,
                num_workers=0,  # Отключаем workers для теста
                shuffle=False,
            )
            batch = next(iter(test_loader))

            # Forward pass
            model.eval()
            with torch.no_grad():
                y = batch.pop("labels")
                X = {k: v.to(device) for k, v in batch.items()}
                y = y.to(device)
                logits = model(**X)

            # Успешно - можно увеличить
            optimal_batch = test_batch
            min_batch = test_batch
            print(f"    ✓ Success")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # OOM - уменьшаем
                max_batch = test_batch
                if device == "cuda":
                    torch.cuda.empty_cache()
                print(f"    ✗ OOM")
            else:
                raise e

    print(f"Optimal batch size found: {optimal_batch}")
    return optimal_batch


def tune_hyperparameters(
    model_config: Dict[str, Any],
    train_dataset: TokenizedDataset,
    val_dataset: TokenizedDataset,
    device: str,
    n_iterations: int = 25,
    n_warmup_epochs: int = 2,
    tuning_sample_size: float = 0.15,
    dropout_range: Tuple[float, float] = (0.1, 0.5),
    lr_range: Tuple[float, float] = (1e-5, 1e-4),
    weight_decay_options: List[float] = [0.0, 0.01, 0.1],
    betas_options: List[Tuple[float, float]] = [(0.9, 0.999), (0.95, 0.999), (0.9, 0.99)],
    tuning_metric: str = "val_loss",
    batch_size: int = 64,
    num_workers: int = 4,
    loss_type: str = "ce",
    metrics_fn: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Random Search для подбора гиперпараметров.

    Args:
        model_config: Конфигурация модели
        train_dataset: Полный train датасет
        val_dataset: Validation датасет для оценки
        device: Устройство
        n_iterations: Количество случайных комбинаций для перебора
        n_warmup_epochs: Количество эпох на каждую комбинацию
        tuning_sample_size: Доля train датасета для перебора (0.15 = 15%)
        dropout_range: Диапазон значений dropout
        lr_range: Диапазон learning rate (логарифмический)
        weight_decay_options: Варианты weight_decay
        betas_options: Варианты betas
        tuning_metric: Метрика для выбора лучших параметров ("val_loss" или "val_accuracy")
        batch_size: Batch size для обучения
        num_workers: Количество workers для DataLoader
        loss_type: Тип loss функции
        metrics_fn: Словарь метрик для вычисления (опционально)

    Returns:
        best_params: Словарь с лучшими параметрами
        results: Список результатов всех итераций
    """
    # Создаем маленькую выборку из train для перебора
    train_size = len(train_dataset)
    sample_size = int(train_size * tuning_sample_size)
    sample_indices = random.sample(range(train_size), sample_size)
    tuning_train_dataset = train_dataset.select(sample_indices)

    print(f"Using {sample_size} samples ({tuning_sample_size*100:.1f}%) from train for tuning")
    print(f"Running {n_iterations} iterations with {n_warmup_epochs} epochs each...")

    best_params = None
    best_score = float("inf") if tuning_metric == "val_loss" else float("-inf")
    results = []

    for iteration in range(n_iterations):
        # Случайная выборка гиперпараметров
        dropout = round(random.uniform(dropout_range[0], dropout_range[1]), 1)

        # LR из логарифмического распределения
        log_lr_min = np.log10(lr_range[0])
        log_lr_max = np.log10(lr_range[1])
        lr = 10 ** random.uniform(log_lr_min, log_lr_max)

        weight_decay = random.choice(weight_decay_options)
        betas = random.choice(betas_options)

        # Пересоздаем модель с новым dropout
        model = ModernBert(
            backbone_name=model_config["model_name"],
            num_classes=model_config["num_classes"],
            dropout=dropout,
            freeze_backbone=model_config.get("freeze_backbone", True),
            use_pooler=model_config.get("use_pooler", False),
            use_float16=model_config.get("use_float16", False),
        )
        model.to(device)

        # Создаем оптимизатор с текущими параметрами
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
        loss_fn = get_loss(loss_type)

        # Создаем dataloaders
        train_loader = build_dataloader(
            tuning_train_dataset,
            model_name=model_config["model_name"],
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        )
        val_loader = build_dataloader(
            val_dataset,
            model_name=model_config["model_name"],
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )

        # Обучаем на маленькой выборке
        model.train()
        for epoch in range(n_warmup_epochs):
            for batch in train_loader:
                y = batch.pop("labels")
                X = {k: v.to(device) for k, v in batch.items()}
                y = y.to(device)

                logits = model(**X)
                loss = loss_fn(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Оцениваем на валидации
        model.eval()
        val_losses = []
        val_probs = []
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                y = batch.pop("labels")
                X = {k: v.to(device) for k, v in batch.items()}
                y = y.to(device)

                logits = model(**X)
                loss = loss_fn(logits, y)

                val_losses.append(loss.item())
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = logits.argmax(-1).cpu().numpy()
                labels = y.cpu().numpy()

                val_probs.extend(probs)
                val_preds.extend(preds)
                val_labels.extend(labels)

        # Вычисляем финальную метрику
        avg_val_loss = np.mean(val_losses)

        # Вычисляем accuracy если нужно
        if tuning_metric == "val_accuracy" and metrics_fn and "accuracy" in metrics_fn:
            score = metrics_fn["accuracy"](
                np.array(val_probs), np.array(val_preds), np.array(val_labels)
            )
        else:
            score = avg_val_loss

        is_better = (
            (score < best_score) if tuning_metric == "val_loss" else (score > best_score)
        )

        results.append(
            {
                "dropout": dropout,
                "lr": lr,
                "weight_decay": weight_decay,
                "betas": betas,
                "score": score,
                "val_loss": avg_val_loss,
            }
        )

        if is_better:
            best_score = score
            best_params = {
                "dropout": dropout,
                "lr": lr,
                "weight_decay": weight_decay,
                "betas": betas,
            }

        print(
            f"  [{iteration+1}/{n_iterations}] dropout={dropout:.1f}, lr={lr:.2e}, "
            f"wd={weight_decay}, betas={betas} -> {tuning_metric}={score:.4f} "
            f"(best={best_score:.4f})"
        )

        # Очищаем память
        del model, optimizer
        if device == "cuda":
            torch.cuda.empty_cache()

    return best_params, results

