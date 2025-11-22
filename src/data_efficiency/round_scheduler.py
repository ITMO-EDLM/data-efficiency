from typing import List, Optional

from torch.utils.data import DataLoader, Subset

from data_efficiency.data import TokenizedDataset
from data_efficiency.strategies.base import DataSelectionStrategy
from data_efficiency.utils.data import build_dataloader


class RoundScheduler:
    """
    Manager of available data for selecting via different selection strategies

    - The run is a full experiment of model training
    - Round is the one epoch where we are selecting data for training.
    After going beyond the rounds budget (the total limit is 10% of the initial dataset),
    the `get_train_dataloader` method will simply return all accumulated training data
    """

    def __init__(
        self,
        run_budget: float,
        rounds_portions: List[float],
        dataset: TokenizedDataset,
        strategy: DataSelectionStrategy,
        model_name: str = "answerdotai/ModernBERT-base",
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        self.rounds_portions = rounds_portions
        self.round_portion_iter = iter(rounds_portions)
        self.dataset = dataset
        self.strategy = strategy
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.N = len(self.dataset)
        self.run_budget = int(self.N * run_budget)
        self.seen = set()
        self.is_reached_budget = False
        self.full_dataloader: Optional[DataLoader] = None

    def _calculate_limit(self) -> int:
        # TO-DO:
        # Maybe should use seen indexes for correctly estimating run budget
        try:
            current_portion = next(self.round_portion_iter)
            return int(self.run_budget * current_portion)
        except StopIteration:
            self.is_reached_budget = True
            self.full_dataloader = self._prepare_dataloader()
            return 0

    def _update_statistics(self, selected_idxs: List[int]) -> None:
        for idx in selected_idxs:
            self.seen.add(idx)

    def _prepare_dataloader(self) -> DataLoader:
        subset = Subset(self.dataset, list(self.seen))
        return build_dataloader(
            subset,
            model_name=self.model_name,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def get_train_dataloader(self, model=None, device: str = "cpu", **kwargs) -> DataLoader:
        limit = self._calculate_limit()
        if self.is_reached_budget:
            return self.full_dataloader
        else:
            # Pass model and device to strategy
            strategy_kwargs = {**kwargs, "model": model, "device": device}
            selected_idxs = self.strategy.select(self.dataset, limit, **strategy_kwargs)
            self._update_statistics(selected_idxs)
            return self._prepare_dataloader()
