from typing import List

from data_efficiency.data import TokenizedDataset


class DataSelectionStrategy:
    def __init__(self):
        pass

    def select(self, dataset: TokenizedDataset, limit: int, **kwargs) -> List[int]:
        """
        Use any methods to extract indexes from dataset for next round

        Args:
            dataset (TokenizedDataset): Slice or all train dataset
            limit (int): Number of objects to extract
            kwargs (Dict[str, Any]): Other required objects for strategy

        Returns:
            selected_indexes (List[int]): Indexes of best objects from input dataset
            for training model
        """
        raise NotImplementedError
