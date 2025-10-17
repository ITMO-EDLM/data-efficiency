from typing import List

import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data_efficiency.data import TokenizedDataset
from data_efficiency.model import ModernBert
from data_efficiency.round_scheduler import RoundScheduler
from data_efficiency.strategies.factory import get_strategy
from data_efficiency.utils.data import build_dataloader


class Trainer:
    def __init__(
        self,
        model: ModernBert,
        val_dataset: TokenizedDataset,
        train_dataset: TokenizedDataset,
        round_budget: float,
        rounds_portions: List[float],
        strategy_type: str,
        lr: float,
        n_epochs: int,
        device: str,
    ):
        self.model = model
        self.val_dataset = val_dataset
        self.train_dataset = train_dataset
        self.round_budget = round_budget
        self.round_portions = rounds_portions
        self.strategy_type = strategy_type
        self.lr = lr
        self.n_epochs = n_epochs
        self.device = device

    def setup(self) -> None:
        """
        Initialize all requirement attributes for run experiment
        """
        self.scheduler = RoundScheduler(
            round_budget=self.round_budget,
            rounds_portions=self.round_portions,
            dataset=self.train_dataset,
            strategy=get_strategy(self.strategy_type),
        )
        self.val_loader = build_dataloader(self.val_dataset, shuffle=False)
        self.optimizer = AdamW(list(self.model.parameters()), lr=self.lr)
        self.model.to(self.device)

    def _train_step(self, train_loader: DataLoader) -> None:
        self.model.train()
        for X, y in tqdm.tqdm(train_loader):
            pass

    def run(self) -> None:
        """
        Running pipeline of training model with some kind of data selection strategy
        """
        for epoch in self.n_epochs:
            print(f"Start {epoch + 1} training epoch")
            train_loader = self.scheduler.get_train_dataloader()
            self._train_step(train_loader)
