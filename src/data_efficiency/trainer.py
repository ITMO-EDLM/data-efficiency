import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

from data_efficiency.data import TokenizedDataset
from data_efficiency.model import ModernBert
from data_efficiency.round_scheduler import RoundScheduler
from data_efficiency.strategies import get_strategy
from data_efficiency.utils import MetricTracker, build_dataloader, get_loss
from data_efficiency.utils.hyperparameter_tuning import (
    find_optimal_batch_size,
    tune_hyperparameters,
)
from data_efficiency.utils.optimizer import create_optimizer_with_different_lr


class Trainer:
    def __init__(
        self,
        model: ModernBert,
        loss_type: str,
        metrics_fn: Dict[str, Callable],
        val_dataset: TokenizedDataset,
        train_dataset: TokenizedDataset,
        run_budget: float,
        rounds_portions: List[float],
        strategy_data: Dict[str, Any],
        optimizer_params: Dict[str, Any],
        n_epochs: int,
        device: str,
        model_name: str = "answerdotai/ModernBERT-base",
        batch_size: int = 64,
        num_workers: int = 4,
        log_dir: str = "./runs",
        run_name: str = None,
        checkpoint_dir: str = "./checkpoints",
        save_checkpoints: bool = True,
        max_checkpoints: int = 3,
        checkpoint_metric: str = "val_loss",
        use_clearml: bool = False,
        clearml_project_name: Optional[str] = None,
        clearml_task_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.loss_type = loss_type
        self.metrics_fn = metrics_fn
        self.val_dataset = val_dataset
        self.train_dataset = train_dataset
        self.run_budget = run_budget
        self.round_portions = rounds_portions
        self.strategy_data = strategy_data
        self.optimizer_params = optimizer_params
        self.n_epochs = n_epochs
        self.device = device
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.log_dir = log_dir
        self.run_name = run_name
        self.checkpoint_dir = checkpoint_dir
        self.save_checkpoints = save_checkpoints
        self.max_checkpoints = max_checkpoints
        self.checkpoint_metric = checkpoint_metric
        self.use_clearml = use_clearml
        self.clearml_project_name = clearml_project_name
        self.clearml_task_name = clearml_task_name
        self.config = config
        self.current_epoch = 0
        self.best_val_metric = None

        # Extract lr_head and lr_backbone from config if present
        self.lr_head = config.get("lr_head") if config else None
        self.lr_backbone = config.get("lr_backbone") if config else None

        # Checkpoint management: list of (metric_value, epoch, path)
        # For loss metrics, lower is better; for accuracy/F1, higher is better
        self.saved_checkpoints: List[Tuple[float, int, Path]] = []

    def setup(self) -> None:
        """
        Initialize all requirement attributes for run experiment
        """
        self.round_scheduler = RoundScheduler(
            run_budget=self.run_budget,
            rounds_portions=self.round_portions,
            dataset=self.train_dataset,
            strategy=get_strategy(**self.strategy_data),
            model_name=self.model_name,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        self.val_loader = build_dataloader(
            self.val_dataset,
            model_name=self.model_name,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
        # Create optimizer with different lr for head and backbone if specified
        self.optimizer = create_optimizer_with_different_lr(
            self.model,
            self.optimizer_params,
            lr_head=self.lr_head,
            lr_backbone=self.lr_backbone,
        )
        self.loss = get_loss(self.loss_type)
        self.model.to(self.device)
        self.tracker = MetricTracker(
            self.metrics_fn,
            log_dir=self.log_dir,
            run_name=self.run_name,
            use_clearml=self.use_clearml,
            clearml_project_name=self.clearml_project_name,
            clearml_task_name=self.clearml_task_name,
        )

        # Log configuration to ClearML if available
        if self.config and self.use_clearml:
            self.tracker.log_configuration(self.config)

    def _train_step(self, train_loader: DataLoader) -> None:
        self.model.train()
        print("Start train round")
        for batch in tqdm.tqdm(train_loader):
            y: torch.Tensor = batch.pop("labels")
            X: Dict[str, torch.Tensor] = batch
            for k, v in X.items():
                X[k] = v.to(self.device)
            y = y.to(self.device)

            logits: torch.Tensor = self.model(**X)
            loss: torch.Tensor = self.loss(logits, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            probs = torch.softmax(logits.detach().cpu(), dim=1).numpy()
            preds = logits.detach().cpu().argmax(-1).numpy()
            labels = y.detach().cpu().numpy()

            self.tracker.save_train_loss(loss)
            self.tracker.save_train_metrics(probs, preds, labels)

    def _val_step(self, val_loader: DataLoader) -> None:
        self.model.eval()
        print("Start validation round")
        for batch in tqdm.tqdm(val_loader):
            y: torch.Tensor = batch.pop("labels")
            X: Dict[str, torch.Tensor] = batch

            for k, v in X.items():
                X[k] = v.to(self.device)
            y = y.to(self.device)

            with torch.set_grad_enabled(False):
                logits: torch.Tensor = self.model(**X)
                loss: torch.Tensor = self.loss(logits, y)

            probs = torch.softmax(logits.detach().cpu(), dim=1).numpy()
            preds = logits.detach().cpu().argmax(-1).numpy()
            labels = y.detach().cpu().numpy()

            self.tracker.save_vall_loss(loss)
            self.tracker.save_val_metrics(probs, preds, labels)

    def _get_epoch_metric(self, epoch: int) -> Optional[float]:
        """Get the metric value for a specific epoch."""
        if self.checkpoint_metric == "val_loss":
            if epoch in self.tracker.val_loss_history:
                values = self.tracker.val_loss_history[epoch]
                if values:  # Check that list is not empty
                    return float(np.mean(values))
        else:
            # For other metrics like accuracy, f1, etc.
            # Remove "val_" prefix if present (e.g., "val_accuracy" -> "accuracy")
            metric_name = self.checkpoint_metric.replace("val_", "")
            if (
                epoch in self.tracker.val_metrics_history
                and metric_name in self.tracker.val_metrics_history[epoch]
            ):
                values = self.tracker.val_metrics_history[epoch][metric_name]
                if values:  # Check that list is not empty
                    return float(np.mean(values))
        return None

    def _is_metric_better(self, new_metric: float, existing_metric: float) -> bool:
        """Check if new metric is better than existing one."""
        # For loss metrics, lower is better; for others (accuracy, f1), higher is better
        if self.checkpoint_metric == "val_loss":
            return new_metric < existing_metric
        else:
            return new_metric > existing_metric

    def save_checkpoint(self, epoch: int) -> None:
        """
        Save model checkpoint only if it's in the top N best models.
        Maintains only the best max_checkpoints models based on checkpoint_metric.
        """
        if not self.save_checkpoints:
            return

        # Get current epoch metric
        current_metric = self._get_epoch_metric(epoch)
        if current_metric is None:
            print(
                f"Warning: Could not get metric for epoch {epoch}, skipping checkpoint save"
            )
            return

        # Create checkpoint directory
        if self.run_name:
            base_dir = Path(self.checkpoint_dir) / self.run_name
        else:
            base_dir = Path(self.checkpoint_dir) / "default_run"

        # Prepare checkpoint data
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss_type": self.loss_type,
            "n_epochs": self.n_epochs,
            "metric_value": current_metric,
            "metric_name": self.checkpoint_metric,
        }

        # Determine if we should save this checkpoint
        should_save = False
        checkpoint_path = None

        if len(self.saved_checkpoints) < self.max_checkpoints:
            # We have space, save it
            should_save = True
            epoch_dir = base_dir / f"epoch_{epoch}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = epoch_dir / "model.pt"
        else:
            # Check if this is better than the worst saved checkpoint
            # Sort checkpoints: for loss (lower is better), for others (higher is better)
            is_loss_metric = self.checkpoint_metric == "val_loss"
            self.saved_checkpoints.sort(key=lambda x: x[0], reverse=not is_loss_metric)

            worst_metric, worst_epoch, worst_path = self.saved_checkpoints[-1]

            if self._is_metric_better(current_metric, worst_metric):
                # Remove worst checkpoint
                if worst_path.exists():
                    worst_path.unlink()
                    # Try to remove parent directory if empty
                    try:
                        worst_path.parent.rmdir()
                    except OSError:
                        pass  # Directory not empty or other error

                # Remove from list
                self.saved_checkpoints.pop()
                should_save = True

                epoch_dir = base_dir / f"epoch_{epoch}"
                epoch_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = epoch_dir / "model.pt"
                print(
                    f"Removed checkpoint from epoch {worst_epoch} (metric: {worst_metric:.4f})"
                )

        if should_save and checkpoint_path:
            torch.save(checkpoint, checkpoint_path)
            self.saved_checkpoints.append((current_metric, epoch, checkpoint_path))

            print(
                f"Checkpoint saved to {checkpoint_path} "
                f"(epoch {epoch}, {self.checkpoint_metric}: {current_metric:.4f})"
            )

            # Update best model if this is the best so far
            is_loss_metric = self.checkpoint_metric == "val_loss"
            self.saved_checkpoints.sort(key=lambda x: x[0], reverse=not is_loss_metric)
            best_metric, best_epoch, _ = self.saved_checkpoints[0]

            # Only update best if current epoch is the best
            if best_epoch == epoch:
                best_dir = base_dir / "best"
                best_dir.mkdir(parents=True, exist_ok=True)
                best_checkpoint_path = best_dir / "model.pt"
                torch.save(checkpoint, best_checkpoint_path)
                print(
                    f"Best model updated (epoch {best_epoch}, "
                    f"{self.checkpoint_metric}: {best_metric:.4f})"
                )
        else:
            print(
                f"Checkpoint not saved (epoch {epoch}, "
                f"{self.checkpoint_metric}: {current_metric:.4f} "
                f"not in top {self.max_checkpoints})"
            )

    def run(self) -> None:
        """
        Running pipeline of training model with some kind of data selection strategy
        """
        for epoch in range(self.n_epochs):
            self.current_epoch = epoch + 1
            # Sync tracker's current_epoch with trainer's current_epoch
            # This ensures metrics are saved to the correct epoch
            self.tracker.current_epoch = self.current_epoch

            print(f"Start {self.current_epoch} epoch")
            train_loader = self.round_scheduler.get_train_dataloader(model=self.model, device=self.device)
            self._train_step(train_loader)
            self._val_step(self.val_loader)

            self.tracker.end_epoch()

            # Save checkpoint only if it's in top N best models
            # Use the epoch number that was just completed
            self.save_checkpoint(self.current_epoch)

        # Close tracker and TensorBoard writer
        self.tracker.close()
        print("Training finished!")

        # Print summary of saved checkpoints
        if self.saved_checkpoints:
            is_loss_metric = self.checkpoint_metric == "val_loss"
            self.saved_checkpoints.sort(key=lambda x: x[0], reverse=not is_loss_metric)
            print(f"\nSaved checkpoints (top {len(self.saved_checkpoints)}):")
            for i, (metric, ep, _) in enumerate(self.saved_checkpoints, 1):
                print(f"  {i}. Epoch {ep}: {self.checkpoint_metric} = {metric:.4f}")

    def warmup(self) -> None:
        """
        Hyperparameter search before main training.
        Only works for "all" strategy.
        """
        print("=" * 60)
        print("Starting Hyperparameter Tuning (Warmup)")
        print("=" * 60)

        # 1. Find optimal batch size
        print("\n[1/2] Finding optimal batch size...")
        optimal_batch_size = find_optimal_batch_size(
            self.model,
            self.train_dataset,
            self.device,
            initial_batch_size=self.batch_size,
        )
        print(f"Optimal batch size: {optimal_batch_size}")
        self.batch_size = optimal_batch_size

        # 2. Hyperparameter search on small sample
        print("\n[2/2] Tuning hyperparameters (Random Search)...")

        # Prepare parameters for search
        tuning_config = self.config or {}
        dropout_range = tuple(tuning_config["dropout_range"])
        lr_range = tuple(tuning_config["lr_range"])
        lr_head_range = (
            tuple(tuning_config["lr_head_range"])
            if tuning_config.get("lr_head_range")
            else None
        )
        lr_backbone_range = (
            tuple(tuning_config["lr_backbone_range"])
            if tuning_config.get("lr_backbone_range")
            else None
        )
        weight_decay_options = tuning_config["weight_decay_options"]
        betas_options_raw = tuning_config["betas_options"]
        betas_options = [tuple(b) for b in betas_options_raw]
        unfreeze_layers_options = tuning_config.get("unfreeze_layers_options")

        # Determine num_classes from config or model
        num_classes = tuning_config.get("num_classes")
        if num_classes is None:
            # Try to determine from classifier output dimension
            num_classes = self.model.classifier.out_features

        model_config = {
            "model_name": self.model_name,
            "num_classes": num_classes,
            "unfreeze_layers": tuning_config.get("unfreeze_layers"),
            "use_pooler": tuning_config.get("use_pooler", False),
            "use_float16": tuning_config.get("use_float16", False),
        }
        train_dataset = self.train_dataset.select(
            random.choices(
                range(len(self.train_dataset)),
                k=int(len(self.train_dataset) * self.run_budget),
            )
        )
        best_params, tuning_results = tune_hyperparameters(
            model_config=model_config,
            train_dataset=train_dataset,
            val_dataset=self.val_dataset,
            device=self.device,
            n_iterations=tuning_config.get("tuning_n_iterations", 25),
            n_warmup_epochs=tuning_config.get("warmup_epochs", 2),
            tuning_sample_size=tuning_config.get("tuning_sample_size", 0.15),
            dropout_range=dropout_range,
            lr_range=lr_range,
            lr_head_range=lr_head_range,
            lr_backbone_range=lr_backbone_range,
            weight_decay_options=weight_decay_options,
            betas_options=betas_options,
            unfreeze_layers_options=unfreeze_layers_options,
            tuning_metric=tuning_config.get("tuning_metric", "val_loss"),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            loss_type=self.loss_type,
            metrics_fn=self.metrics_fn,
        )

        print("\nBest hyperparameters found:")
        print(f"  dropout: {best_params['dropout']}")
        print(
            f"  lr_head: {best_params.get('lr_head', best_params.get('lr', 'N/A')):.2e}"
        )
        if best_params.get("lr_backbone") is not None:
            print(f"  lr_backbone: {best_params['lr_backbone']:.2e}")
        if best_params.get("unfreeze_layers") is not None:
            print(f"  unfreeze_layers: {best_params['unfreeze_layers']}")
        print(f"  weight_decay: {best_params['weight_decay']}")
        print(f"  betas: {best_params['betas']}")

        # 3. Update configuration
        # For backward compatibility
        base_lr = best_params.get("lr_head", best_params.get("lr", 2e-5))
        self.optimizer_params = {
            "lr": base_lr,
            "weight_decay": best_params["weight_decay"],
            "betas": best_params["betas"],
        }

        # Update lr_head and lr_backbone in config
        if best_params.get("lr_head") is not None:
            self.lr_head = best_params["lr_head"]
        if best_params.get("lr_backbone") is not None:
            self.lr_backbone = best_params["lr_backbone"]

        # 4. Recreate model with optimal dropout and unfreeze_layers
        print("\nRecreating model with optimal dropout and resetting weights...")
        self.model = ModernBert(
            backbone_name=self.model_name,
            num_classes=model_config["num_classes"],
            dropout=best_params["dropout"],
            unfreeze_layers=best_params.get("unfreeze_layers"),
            use_pooler=model_config["use_pooler"],
            use_float16=model_config["use_float16"],
        )
        self.model.to(self.device)

        # 5. Recreate optimizer with new parameters
        # Use lr_head and lr_backbone from best_params if available
        best_lr_head = best_params.get("lr_head")
        best_lr_backbone = best_params.get("lr_backbone")
        self.optimizer = create_optimizer_with_different_lr(
            self.model,
            self.optimizer_params,
            lr_head=best_lr_head,
            lr_backbone=best_lr_backbone,
        )

        # 6. Update round_scheduler with new batch_size
        self.round_scheduler = RoundScheduler(
            run_budget=self.run_budget,
            rounds_portions=self.round_portions,
            dataset=self.train_dataset,
            strategy=get_strategy(**self.strategy_data),
            model_name=self.model_name,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        # 7. Update val_loader with new batch_size
        self.val_loader = build_dataloader(
            self.val_dataset,
            model_name=self.model_name,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

        print("=" * 60)
        print("Hyperparameter Tuning Complete")
        print("=" * 60)
