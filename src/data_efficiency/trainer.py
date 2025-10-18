from typing import Any, Callable, Dict, List

import torch
import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data_efficiency.data import TokenizedDataset
from data_efficiency.model import ModernBert
from data_efficiency.round_scheduler import RoundScheduler
from data_efficiency.strategies import get_strategy
from data_efficiency.utils import MetricTracker, build_dataloader, get_loss


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
        log_dir: str = "./runs",
        run_name: str = None,
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
        self.log_dir = log_dir
        self.run_name = run_name

    def setup(self) -> None:
        """
        Initialize all requirement attributes for run experiment
        """
        self.round_scheduler = RoundScheduler(
            run_budget=self.run_budget,
            rounds_portions=self.round_portions,
            dataset=self.train_dataset,
            strategy=get_strategy(**self.strategy_data),
        )
        self.val_loader = build_dataloader(self.val_dataset, shuffle=False)
        self.optimizer = AdamW(list(self.model.parameters()), **self.optimizer_params)
        self.loss = get_loss(self.loss_type)
        self.model.to(self.device)
        self.tracker = MetricTracker(self.metrics_fn, log_dir=self.log_dir, run_name=self.run_name)

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

    def run(self) -> None:
        """
        Running pipeline of training model with some kind of data selection strategy
        """
        for epoch in range(self.n_epochs):
            print(f"Start {epoch + 1} epoch")
            train_loader = self.round_scheduler.get_train_dataloader()
            self._train_step(train_loader)
            self._val_step(self.val_loader)
            self.tracker.end_epoch()

        # Close tracker and TensorBoard writer
        self.tracker.close()
        print("Training finished!")
