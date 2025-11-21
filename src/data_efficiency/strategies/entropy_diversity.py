"""Entropy + Diversity strategy combining model uncertainty with geometric diversity."""

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

        # Compute entropy for all samples (on GPU)
        entropy = compute_entropy(probs)  # Returns torch tensor

        # Sort indices by entropy (descending)
        sorted_indices = torch.argsort(entropy, descending=True).cpu().numpy()

        selected_indices = []
        remaining_candidates = list(sorted_indices)

        while len(selected_indices) < limit and len(remaining_candidates) > 0:
            if len(selected_indices) == 0:
                # First sample: always add
                selected_indices.append(int(remaining_candidates[0]))
                remaining_candidates.pop(0)
            else:
                # Get embeddings for selected and all remaining candidate samples
                selected_embeddings = embeddings[selected_indices]  # Shape: (n_selected, dim)
                candidate_embeddings = embeddings[
                    remaining_candidates
                ]  # Shape: (n_candidates, dim)

                # Compute minimum distances for all candidates at once (GPU-accelerated)
                min_distances = compute_min_distances_gpu(
                    candidate_embeddings, selected_embeddings
                )  # Shape: (n_candidates,)

                # Find first candidate that satisfies threshold
                valid_mask = min_distances > self.threshold
                valid_indices = torch.where(valid_mask)[0]

                if len(valid_indices) > 0:
                    # Take first valid candidate (they're already sorted by entropy)
                    candidate_idx = valid_indices[0].item()
                    selected_indices.append(int(remaining_candidates[candidate_idx]))
                    remaining_candidates.pop(candidate_idx)
                else:
                    # No more candidates satisfy the threshold, break
                    break

        return selected_indices
