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
    ):
        self.rounds_portions = rounds_portions
        self.round_portion_iter = iter(rounds_portions)
        self.dataset = dataset
        self.strategy = strategy

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
        subset = Subset(self.dataset, self.seen)
        return build_dataloader(subset, shuffle=True)

    def get_train_dataloader(self, **kwargs) -> DataLoader:
        limit = self._calculate_limit()
        if self.is_reached_budget:
            return self.full_dataloader
        else:
            selected_idxs = self.strategy.select(self.dataset, limit, **kwargs)
            self._update_statistics(selected_idxs)
            return self._prepare_dataloader()
