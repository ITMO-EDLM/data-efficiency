"""K-Center Greedy strategy for geometric coreset sampling."""

import random
from typing import List

import numpy as np

from data_efficiency.data import TokenizedDataset
from data_efficiency.strategies.base import DataSelectionStrategy
from data_efficiency.utils.embeddings import get_embeddings


class KCenterGreedyStrategy(DataSelectionStrategy):
    """
    K-Center Greedy strategy for geometric coreset sampling.

    Maximizes the minimum distance between new elements and already selected ones.
    """

    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-base",
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        """
        Initialize K-Center Greedy strategy.

        Args:
            model_name: Model name for computing embeddings
            batch_size: Batch size for embedding computation
            num_workers: Number of workers for data loading
        """
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_workers = num_workers

    def select(
        self, dataset: TokenizedDataset, limit: int, model=None, device: str = "cpu", **kwargs
    ) -> List[int]:
        """
        Select samples using K-Center Greedy algorithm.

        Args:
            dataset: Dataset to select from
            limit: Number of samples to select
            model: Model for computing embeddings (required)
            device: Device to run computation on
            **kwargs: Additional arguments

        Returns:
            List of selected indices
        """
        if model is None:
            raise ValueError("Model is required for K-Center Greedy strategy")

        if limit <= 0:
            return []

        n_samples = len(dataset)
        if limit >= n_samples:
            return list(range(n_samples))

        # Compute embeddings for all samples
        embeddings = get_embeddings(
            model,
            dataset,
            device,
            model_name=self.model_name,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        # Initialize: randomly select first sample
        selected_indices = [random.randint(0, n_samples - 1)]
        remaining_indices = set(range(n_samples)) - set(selected_indices)

        # Iteratively add samples that maximize minimum distance
        while len(selected_indices) < limit:
            selected_embeddings = embeddings[selected_indices]

            # For each remaining sample, compute minimum distance to selected samples
            max_min_dist = -1
            best_idx = None

            for idx in remaining_indices:
                # Compute distance to all selected samples
                distances = np.linalg.norm(embeddings[idx] - selected_embeddings, axis=1)
                min_dist = np.min(distances)

                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            else:
                # Fallback: add random sample if no improvement found
                if remaining_indices:
                    random_idx = random.choice(list(remaining_indices))
                    selected_indices.append(random_idx)
                    remaining_indices.remove(random_idx)
                else:
                    break

        return selected_indices
