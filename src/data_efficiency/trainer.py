import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import tqdm
from torch.optim import AdamW
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
        self.use_clearml = use_clearml
        self.clearml_project_name = clearml_project_name
        self.clearml_task_name = clearml_task_name
        self.config = config
        self.current_epoch = 0
        self.best_val_metric = None

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
        self.optimizer = AdamW(list(self.model.parameters()), **self.optimizer_params)
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

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint with metadata."""
        if not self.save_checkpoints:
            return

        # Create checkpoint directory
        if self.run_name:
            base_dir = Path(self.checkpoint_dir) / self.run_name
        else:
            base_dir = Path(self.checkpoint_dir) / "default_run"

        # Save epoch checkpoint
        epoch_dir = base_dir / f"epoch_{epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss_type": self.loss_type,
            "n_epochs": self.n_epochs,
        }

        checkpoint_path = epoch_dir / "model.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

        # Save as best model if applicable
        if is_best:
            best_dir = base_dir / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            best_path = best_dir / "model.pt"
            torch.save(checkpoint, best_path)
            print(f"Best model saved to {best_path}")

    def run(self) -> None:
        """
        Running pipeline of training model with some kind of data selection strategy
        """
        for epoch in range(self.n_epochs):
            self.current_epoch = epoch + 1
            print(f"Start {self.current_epoch} epoch")
            train_loader = self.round_scheduler.get_train_dataloader()
            self._train_step(train_loader)
            self._val_step(self.val_loader)
            self.tracker.end_epoch()

            # Save checkpoint after each epoch
            # Determine if this is the best model based on validation metrics
            # For simplicity, we'll save the last epoch as best
            is_best = epoch == self.n_epochs - 1
            self.save_checkpoint(self.current_epoch, is_best=is_best)

        # Close tracker and TensorBoard writer
        self.tracker.close()
        print("Training finished!")

    def warmup(self) -> None:
        """
        Перебор гиперпараметров перед основным обучением.
        Работает только для стратегии "all".
        """
        print("=" * 60)
        print("Starting Hyperparameter Tuning (Warmup)")
        print("=" * 60)

        # 1. Определяем оптимальный batch size
        print("\n[1/2] Finding optimal batch size...")
        optimal_batch_size = find_optimal_batch_size(
            self.model, self.train_dataset, self.device, initial_batch_size=self.batch_size
        )
        print(f"Optimal batch size: {optimal_batch_size}")
        self.batch_size = optimal_batch_size

        # 2. Перебор гиперпараметров на маленькой выборке
        print("\n[2/2] Tuning hyperparameters (Random Search)...")

        # Подготавливаем параметры для перебора
        tuning_config = self.config or {}
        dropout_range = tuple(tuning_config["dropout_range"])
        lr_range = tuple(tuning_config["lr_range"])
        weight_decay_options = tuning_config["weight_decay_options"]
        betas_options_raw = tuning_config["betas_options"]
        betas_options = [tuple(b) for b in betas_options_raw]

        # Определяем num_classes из конфига или из модели
        num_classes = tuning_config.get("num_classes")
        if num_classes is None:
            # Пытаемся определить из размерности классификатора
            num_classes = self.model.classifier.out_features

        model_config = {
            "model_name": self.model_name,
            "num_classes": num_classes,
            "freeze_backbone": tuning_config.get("freeze_backbone", True),
            "use_pooler": tuning_config.get("use_pooler", False),
            "use_float16": tuning_config.get("use_float16", False),
        }
        train_dataset = self.train_dataset.select(
            random.choices(
                range(len(self.train_dataset)), k=int(len(self.train_dataset) * self.run_budget)
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
            weight_decay_options=weight_decay_options,
            betas_options=betas_options,
            tuning_metric=tuning_config.get("tuning_metric", "val_loss"),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            loss_type=self.loss_type,
            metrics_fn=self.metrics_fn,
        )

        print("\nBest hyperparameters found:")
        print(f"  dropout: {best_params['dropout']}")
        print(f"  lr: {best_params['lr']:.2e}")
        print(f"  weight_decay: {best_params['weight_decay']}")
        print(f"  betas: {best_params['betas']}")

        # 3. Обновляем конфигурацию
        self.optimizer_params = {
            "lr": best_params["lr"],
            "weight_decay": best_params["weight_decay"],
            "betas": best_params["betas"],
        }

        # 4. Пересоздаем модель с оптимальным dropout и сбрасываем веса
        print("\nRecreating model with optimal dropout and resetting weights...")
        self.model = ModernBert(
            backbone_name=self.model_name,
            num_classes=model_config["num_classes"],
            dropout=best_params["dropout"],
            freeze_backbone=model_config["freeze_backbone"],
            use_pooler=model_config["use_pooler"],
            use_float16=model_config["use_float16"],
        )
        self.model.to(self.device)

        # 5. Пересоздаем оптимизатор с новыми параметрами
        self.optimizer = AdamW(self.model.parameters(), **self.optimizer_params)

        # 6. Обновляем round_scheduler с новым batch_size
        self.round_scheduler = RoundScheduler(
            run_budget=self.run_budget,
            rounds_portions=self.round_portions,
            dataset=self.train_dataset,
            strategy=get_strategy(**self.strategy_data),
            model_name=self.model_name,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        # 7. Обновляем val_loader с новым batch_size
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
