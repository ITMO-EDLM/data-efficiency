"""QDIT-lite strategy: Quality + Diversity hybrid."""

import random
from typing import List

import torch

from data_efficiency.data import TokenizedDataset
from data_efficiency.strategies.base import DataSelectionStrategy
from data_efficiency.utils.embeddings import (
    compute_entropy,
    compute_min_distances_gpu,
    get_embeddings,
    get_predictions,
)


class QDITLiteStrategy(DataSelectionStrategy):
    """
    QDIT-lite strategy: Quality + Diversity hybrid.

    Combines quality proxy (1 - entropy) with geometric diversity.
    Score = alpha * Q(x) + (1 - alpha) * D(x)
    """

    def __init__(
        self,
        alpha: float = 0.5,
        model_name: str = "answerdotai/ModernBERT-base",
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        """
        Initialize QDIT-lite strategy.

        Args:
            alpha: Weight for quality term (0-1). Higher alpha = more weight on quality.
            model_name: Model name for computing embeddings and predictions
            batch_size: Batch size for processing
            num_workers: Number of workers for data loading
        """
        super().__init__()
        if not 0 <= alpha <= 1:
            raise ValueError("Alpha must be between 0 and 1")
        self.alpha = alpha
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_workers = num_workers

    def select(
        self, dataset: TokenizedDataset, limit: int, model=None, device: str = "cpu", **kwargs
    ) -> List[int]:
        """
        Select samples using QDIT-lite approach.

        Args:
            dataset: Dataset to select from
            limit: Number of samples to select
            model: Model for computing embeddings and predictions (required)
            device: Device to run computation on
            **kwargs: Additional arguments

        Returns:
            List of selected indices
        """
        if model is None:
            raise ValueError("Model is required for QDIT-lite strategy")

        if limit <= 0:
            return []

        n_samples = len(dataset)
        if limit >= n_samples:
            return list(range(n_samples))

        # Compute predictions and embeddings (keep on GPU)
        probs = get_predictions(
            model,
            dataset,
            device,
            model_name=self.model_name,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            return_numpy=False,  # Keep as torch tensor on GPU
        )
        embeddings = get_embeddings(
            model,
            dataset,
            device,
            model_name=self.model_name,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            return_numpy=False,  # Keep as torch tensor on GPU
        )

        # Compute quality proxy: Q(x) = 1 - H(x) (on GPU)
        entropy = compute_entropy(probs)  # Returns torch tensor
        quality = 1.0 - entropy  # Shape: (n_samples,)

        # Initialize: randomly select first sample
        selected_indices = [random.randint(0, n_samples - 1)]
        remaining_indices = list(set(range(n_samples)) - set(selected_indices))

        # Iteratively add samples that maximize combined score
        while len(selected_indices) < limit:
            if len(remaining_indices) == 0:
                break

            # Get embeddings for selected and remaining samples
            selected_embeddings = embeddings[selected_indices]  # Shape: (n_selected, dim)
            remaining_embeddings = embeddings[remaining_indices]  # Shape: (n_remaining, dim)

            # Compute diversity term for all remaining samples at once (GPU-accelerated)
            # D(x) = min distance to selected samples
            min_distances = compute_min_distances_gpu(
                remaining_embeddings, selected_embeddings
            )  # Shape: (n_remaining,)

            # Get quality scores for remaining samples
            remaining_quality = quality[remaining_indices]  # Shape: (n_remaining,)

            # Combined score: alpha * Q(x) + (1 - alpha) * D(x)
            scores = self.alpha * remaining_quality + (1 - self.alpha) * min_distances

            # Find the index with maximum score
            best_local_idx = torch.argmax(scores).item()
            best_idx = remaining_indices[best_local_idx]

            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        return selected_indices
