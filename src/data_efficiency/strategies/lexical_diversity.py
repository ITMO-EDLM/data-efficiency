"""Lexical Diversity strategy using HD-D, MTLD, or TTR metrics."""

from typing import List

import numpy as np

from data_efficiency.data import TokenizedDataset
from data_efficiency.strategies.base import DataSelectionStrategy
from data_efficiency.utils.lexical_diversity import compute_lexical_diversity


class LexicalDiversityStrategy(DataSelectionStrategy):
    """
    Lexical Diversity strategy.

    Selects samples based on lexical diversity metrics (HD-D, MTLD, or TTR).
    """

    def __init__(self, metric: str = "hdd"):
        """
        Initialize Lexical Diversity strategy.

        Args:
            metric: Lexical diversity metric to use ('hdd', 'mtld', or 'ttr')
        """
        super().__init__()
        if metric not in ["hdd", "mtld", "ttr"]:
            raise ValueError(f"Unknown metric: {metric}. Choose from 'hdd', 'mtld', 'ttr'")
        self.metric = metric

    def select(self, dataset: TokenizedDataset, limit: int, **kwargs) -> List[int]:
        """
        Select samples based on lexical diversity.

        Args:
            dataset: Dataset to select from
            limit: Number of samples to select
            **kwargs: Additional arguments (not used)

        Returns:
            List of selected indices
        """
        if limit <= 0:
            return []

        n_samples = len(dataset)
        if limit >= n_samples:
            return list(range(n_samples))

        # Compute lexical diversity for all samples
        diversity_scores = []
        for i in range(n_samples):
            item = dataset[i]
            # Get text from dataset item
            text = item.get("sentence", "")
            if isinstance(text, list):
                text = " ".join(text)
            score = compute_lexical_diversity(text, metric=self.metric)
            diversity_scores.append(score)

        diversity_scores = np.array(diversity_scores)

        # Sort by diversity score (descending) and take top limit
        sorted_indices = np.argsort(diversity_scores)[::-1]
        selected_indices = sorted_indices[:limit].tolist()

        return [int(idx) for idx in selected_indices]
