"""Entropy + Diversity strategy combining model uncertainty with geometric diversity."""

from typing import List

import numpy as np

from data_efficiency.data import TokenizedDataset
from data_efficiency.strategies.base import DataSelectionStrategy
from data_efficiency.utils.embeddings import compute_entropy, get_embeddings, get_predictions


class EntropyDiversityStrategy(DataSelectionStrategy):
    """
    Entropy + Diversity strategy.

    Selects samples with high uncertainty (entropy) while maintaining
    diversity by enforcing minimum distance threshold.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        model_name: str = "answerdotai/ModernBERT-base",
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        """
        Initialize Entropy + Diversity strategy.

        Args:
            threshold: Minimum distance threshold for diversity (tau in spec)
            model_name: Model name for computing embeddings and predictions
            batch_size: Batch size for processing
            num_workers: Number of workers for data loading
        """
        super().__init__()
        self.threshold = threshold
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_workers = num_workers

    def select(
        self, dataset: TokenizedDataset, limit: int, model=None, device: str = "cpu", **kwargs
    ) -> List[int]:
        """
        Select samples using entropy + diversity approach.

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
            raise ValueError("Model is required for Entropy + Diversity strategy")

        if limit <= 0:
            return []

        n_samples = len(dataset)
        if limit >= n_samples:
            return list(range(n_samples))

        # Compute predictions and embeddings
        probs = get_predictions(
            model,
            dataset,
            device,
            model_name=self.model_name,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        embeddings = get_embeddings(
            model,
            dataset,
            device,
            model_name=self.model_name,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        # Compute entropy for all samples
        entropy = compute_entropy(probs)

        # Sort indices by entropy (descending)
        sorted_indices = np.argsort(entropy)[::-1]

        selected_indices = []
        for idx in sorted_indices:
            if len(selected_indices) >= limit:
                break

            if len(selected_indices) == 0:
                # First sample: always add
                selected_indices.append(int(idx))
            else:
                # Check diversity constraint: min distance to selected samples > threshold
                selected_embeddings = embeddings[selected_indices]
                distances = np.linalg.norm(embeddings[idx] - selected_embeddings, axis=1)
                min_dist = np.min(distances)

                if min_dist > self.threshold:
                    selected_indices.append(int(idx))

        return selected_indices
