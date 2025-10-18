import random

from data_efficiency.data import TokenizedDataset
from data_efficiency.strategies.base import DataSelectionStrategy


class RandomDatasetSelectionStrategy(DataSelectionStrategy):
    def select(self, dataset: TokenizedDataset, limit: int):
        """
        Return random dataset limit size.

        Args:
            dataset (TokenizedDataset): All train or validation dataset
            limit (int): number of objects to extract

        Returns:
            selected_indexes (List[int]): Indexes of random dataset objects
        """
        selected_indexes = random.sample(range(len(dataset)), limit)
        return selected_indexes
