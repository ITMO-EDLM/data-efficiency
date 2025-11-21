"""K-Center Greedy strategy for geometric coreset sampling."""

import random
from typing import List

import torch

from data_efficiency.data import TokenizedDataset
from data_efficiency.strategies.base import DataSelectionStrategy
from data_efficiency.utils.embeddings import (
    compute_min_distances_gpu,
    get_embeddings,
)


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

        # Compute embeddings for all samples (keep on GPU)
        embeddings = get_embeddings(
            model,
            dataset,
            device,
            model_name=self.model_name,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            return_numpy=False,  # Keep as torch tensor on GPU
        )

        # Initialize: randomly select first sample
        selected_indices = [random.randint(0, n_samples - 1)]
        remaining_indices = list(set(range(n_samples)) - set(selected_indices))

        # Iteratively add samples that maximize minimum distance
        while len(selected_indices) < limit:
            if len(remaining_indices) == 0:
                break

            # Get embeddings for selected and remaining samples
            selected_embeddings = embeddings[selected_indices]  # Shape: (n_selected, dim)
            remaining_embeddings = embeddings[remaining_indices]  # Shape: (n_remaining, dim)

            # Compute minimum distances for all remaining samples at once (GPU-accelerated)
            min_distances = compute_min_distances_gpu(
                remaining_embeddings, selected_embeddings
            )  # Shape: (n_remaining,)

            # Find the index with maximum minimum distance
            max_min_dist_idx = torch.argmax(min_distances).item()
            best_idx = remaining_indices[max_min_dist_idx]

            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        return selected_indices
