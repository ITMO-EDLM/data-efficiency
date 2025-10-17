from data_efficiency.data import TokenizedDataset
from data_efficiency.strategies.base import DataSelectionStrategy


class AllDatasetSelectionStrategy(DataSelectionStrategy):
    def select(self, dataset: TokenizedDataset, limit: int):
        """
        Return all provided dataset. Backward for training model on full dataset.

        Args:
            dataset (TokenizedDataset): All train dataset
            limit (int): number of objects to extract

        Returns:
            selected_indexes (List[int]): Indexes of all dataset objects
        """
        selected_indexes = [int(item["idx"].item()) for item in dataset]
        return selected_indexes
