"""QDIT-lite strategy: Quality + Diversity hybrid."""

import random
from typing import List

import numpy as np

from data_efficiency.data import TokenizedDataset
from data_efficiency.strategies.base import DataSelectionStrategy
from data_efficiency.utils.embeddings import compute_entropy, get_embeddings, get_predictions


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

        # Compute quality proxy: Q(x) = 1 - H(x)
        entropy = compute_entropy(probs)
        quality = 1.0 - entropy

        # Initialize: randomly select first sample
        selected_indices = [random.randint(0, n_samples - 1)]
        remaining_indices = set(range(n_samples)) - set(selected_indices)

        # Iteratively add samples that maximize combined score
        while len(selected_indices) < limit:
            selected_embeddings = embeddings[selected_indices]

            best_score = -np.inf
            best_idx = None

            for idx in remaining_indices:
                # Quality term: Q(x)
                q = quality[idx]

                # Diversity term: D(x) = min distance to selected samples
                distances = np.linalg.norm(embeddings[idx] - selected_embeddings, axis=1)
                d = np.min(distances)

                # Combined score
                score = self.alpha * q + (1 - self.alpha) * d

                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            else:
                # Fallback: add random sample
                if remaining_indices:
                    random_idx = random.choice(list(remaining_indices))
                    selected_indices.append(random_idx)
                    remaining_indices.remove(random_idx)
                else:
                    break

        return selected_indices
