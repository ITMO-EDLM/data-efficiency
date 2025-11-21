import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import tqdm

from data_efficiency.data import TokenizedDataset
from data_efficiency.model import ModernBert
from data_efficiency.utils.data import build_dataloader
from data_efficiency.utils.loss import get_loss
from data_efficiency.utils.optimizer import create_optimizer_with_different_lr


def find_optimal_batch_size(
    model: ModernBert,
    dataset: TokenizedDataset,
    device: str,
    initial_batch_size: int = 64,
    max_iterations: int = 10,
) -> int:
    """
    Binary search for maximum batch size without OOM.

    Args:
        model: Model for testing
        dataset: Dataset for testing
        device: Device (cuda/mps/cpu)
        initial_batch_size: Initial batch size
        max_iterations: Maximum number of search iterations

    Returns:
        optimal_batch_size: Found optimal batch size
    """
    min_batch = 1
    max_batch = initial_batch_size * 8
    optimal_batch = initial_batch_size

    print(
        f"Searching for optimal batch size (initial: {initial_batch_size}, max: {max_batch})..."
    )

    for _ in tqdm.tqdm(range(max_iterations)):
        if max_batch - min_batch < 2:
            break

        test_batch = (min_batch + max_batch) // 2
        print(f"  Testing batch size: {test_batch} (range: {min_batch}-{max_batch})")

        try:
            # Create small dataloader for test
            test_loader = build_dataloader(
                dataset,
                batch_size=test_batch,
                num_workers=0,  # Disable workers for test
                shuffle=False,
            )
            batch = next(iter(test_loader))

            # Forward pass
            model.eval()
            with torch.no_grad():
                y = batch.pop("labels")
                X = {k: v.to(device) for k, v in batch.items()}
                y = y.to(device)
                _ = model(**X)

            # Success - can increase
            optimal_batch = test_batch
            min_batch = test_batch
            print("    ✓ Success")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # OOM - decrease
                max_batch = test_batch
                if device == "cuda":
                    torch.cuda.empty_cache()
                print("    ✗ OOM")
            else:
                raise e

    print(f"Optimal batch size found: {optimal_batch}")
    return optimal_batch


def tune_hyperparameters(
    model_config: Dict[str, Any],
    train_dataset: TokenizedDataset,
    val_dataset: TokenizedDataset,
    device: str,
    n_iterations: int = 2,
    n_warmup_epochs: int = 2,
    tuning_sample_size: float = 0.15,
    dropout_range: Tuple[float, float] = (0.1, 0.5),
    lr_range: Tuple[float, float] = (1e-5, 1e-3),
    lr_head_range: Optional[Tuple[float, float]] = None,
    lr_backbone_range: Optional[Tuple[float, float]] = None,
    weight_decay_options: List[float] = [0.0, 0.01, 0.1],  # noqa: B006
    betas_options: List[Tuple[float, float]] = [
        (0.9, 0.999),
        (0.95, 0.999),
        (0.9, 0.99),
    ],  # noqa: B006
    unfreeze_layers_options: Optional[List[int]] = None,
    tuning_metric: str = "val_loss",
    batch_size: int = 64,
    num_workers: int = 4,
    loss_type: str = "ce",
    metrics_fn: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Random Search for hyperparameter tuning.

    Args:
        model_config: Model configuration
        train_dataset: Full train dataset
        val_dataset: Validation dataset for evaluation
        device: Device
        n_iterations: Number of random combinations to search
        n_warmup_epochs: Number of epochs for each combination
        tuning_sample_size: Fraction of train dataset for search (0.15 = 15%)
        dropout_range: Dropout value range
        lr_range: Learning rate range (logarithmic) - used if lr_head_range
         and lr_backbone_range are not specified
        lr_head_range: Learning rate range for head (if None, uses lr_range)
        lr_backbone_range: Learning rate range for backbone (if None, uses lr_range)
        weight_decay_options: Weight decay options
        betas_options: Betas options
        unfreeze_layers_options: Options for number of unfrozen layers
         (if None, uses value from model_config)
        tuning_metric: Metric for selecting best parameters ("val_loss" or "val_accuracy")
        batch_size: Batch size for training
        num_workers: Number of workers for DataLoader
        loss_type: Loss function type
        metrics_fn: Dictionary of metrics to compute (optional)

    Returns:
        best_params: Dictionary with best parameters
        results: List of results from all iterations
    """
    # Create small sample from train for search
    train_size = len(train_dataset)
    sample_size = int(train_size * tuning_sample_size)
    sample_indices = random.sample(range(train_size), sample_size)
    tuning_train_dataset = train_dataset.select(sample_indices)

    print(
        f"Using {sample_size} samples ({tuning_sample_size * 100:.1f}%) from train for tuning"
    )
    print(f"Running {n_iterations} iterations with {n_warmup_epochs} epochs each...")

    best_params = None
    best_score = float("inf") if tuning_metric == "val_loss" else float("-inf")
    results = []

    for iteration in range(n_iterations):
        print(f"Start {iteration + 1} iteration\n\n")
        # Random hyperparameter sampling
        dropout = round(random.uniform(dropout_range[0], dropout_range[1]), 1)

        # Determine number of unfrozen layers
        unfreeze_layers = None
        if unfreeze_layers_options is not None and len(unfreeze_layers_options) > 0:
            unfreeze_layers = random.choice(unfreeze_layers_options)
        elif model_config.get("unfreeze_layers") is not None:
            unfreeze_layers = model_config.get("unfreeze_layers")

        # LR from logarithmic distribution
        # If separate ranges for head and backbone are specified, use them
        if lr_head_range is not None:
            log_lr_head_min = np.log10(lr_head_range[0])
            log_lr_head_max = np.log10(lr_head_range[1])
            lr_head = 10 ** random.uniform(log_lr_head_min, log_lr_head_max)
        else:
            log_lr_min = np.log10(lr_range[0])
            log_lr_max = np.log10(lr_range[1])
            lr_head = 10 ** random.uniform(log_lr_min, log_lr_max)

        if lr_backbone_range is not None:
            log_lr_backbone_min = np.log10(lr_backbone_range[0])
            log_lr_backbone_max = np.log10(lr_backbone_range[1])
            lr_backbone = 10 ** random.uniform(log_lr_backbone_min, log_lr_backbone_max)
        else:
            log_lr_min = np.log10(lr_range[0])
            log_lr_max = np.log10(lr_range[1])
            lr_backbone = 10 ** random.uniform(log_lr_min, log_lr_max)

        weight_decay = random.choice(weight_decay_options)
        betas = random.choice(betas_options)

        # Recreate model with new dropout and unfreeze_layers
        model = ModernBert(
            backbone_name=model_config["model_name"],
            num_classes=model_config["num_classes"],
            dropout=dropout,
            unfreeze_layers=unfreeze_layers,
            use_pooler=model_config.get("use_pooler", False),
            use_float16=model_config.get("use_float16", False),
        )
        model.to(device)

        # Create optimizer with different lr for head and backbone
        optimizer_params = {
            "lr": lr_head,  # Base lr (will be overridden for groups)
            "weight_decay": weight_decay,
            "betas": betas,
        }
        optimizer = create_optimizer_with_different_lr(
            model,
            optimizer_params,
            lr_head=lr_head,
            lr_backbone=(
                lr_backbone if unfreeze_layers and unfreeze_layers > 0 else None
            ),
        )
        loss_fn = get_loss(loss_type)

        # Create dataloaders
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

        # Train on small sample
        model.train()
        for epoch in range(n_warmup_epochs):
            print(f"Start {epoch + 1} train epoch")
            for batch in tqdm.tqdm(train_loader):
                y = batch.pop("labels")
                X = {k: v.to(device) for k, v in batch.items()}
                y = y.to(device)

                logits = model(**X)
                loss = loss_fn(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Evaluate on validation
        model.eval()
        val_losses = []
        val_probs = []
        val_preds = []
        val_labels = []

        with torch.no_grad():
            print("Start validation")
            for batch in tqdm.tqdm(val_loader):
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

        # Compute final metric
        avg_val_loss = np.mean(val_losses)

        # Compute accuracy if needed
        if tuning_metric == "val_accuracy" and metrics_fn and "accuracy" in metrics_fn:
            score = metrics_fn["accuracy"](
                np.array(val_probs), np.array(val_preds), np.array(val_labels)
            )
        else:
            score = avg_val_loss

        is_better = (
            (score < best_score)
            if tuning_metric == "val_loss"
            else (score > best_score)
        )

        result_dict = {
            "dropout": dropout,
            "lr_head": lr_head,
            "lr_backbone": (
                lr_backbone if unfreeze_layers and unfreeze_layers > 0 else None
            ),
            "unfreeze_layers": unfreeze_layers,
            "weight_decay": weight_decay,
            "betas": betas,
            "score": score,
            "val_loss": avg_val_loss,
        }
        results.append(result_dict)

        if is_better:
            best_score = score
            best_params = {
                "dropout": dropout,
                "lr_head": lr_head,
                "lr_backbone": (
                    lr_backbone if unfreeze_layers and unfreeze_layers > 0 else None
                ),
                "unfreeze_layers": unfreeze_layers,
                "weight_decay": weight_decay,
                "betas": betas,
            }

        unfreeze_info = (
            f", unfreeze_layers={unfreeze_layers}" if unfreeze_layers else ""
        )
        lr_info = f"lr_head={lr_head:.2e}"
        if unfreeze_layers and unfreeze_layers > 0:
            lr_info += f", lr_backbone={lr_backbone:.2e}"
        print(
            f"  [{iteration + 1}/{n_iterations}] dropout={dropout:.1f}, {lr_info}, "
            f"wd={weight_decay}, betas={betas}{unfreeze_info} -> {tuning_metric}={score:.4f} "
            f"(best={best_score:.4f})"
        )

        # Clear memory
        del model, optimizer
        if device == "cuda":
            torch.cuda.empty_cache()

    return best_params, results
