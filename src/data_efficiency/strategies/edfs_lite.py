"""EDFS-lite strategy: Easy-and-Diverse-First."""

from typing import List, Tuple

import torch

from data_efficiency.data import TokenizedDataset
from data_efficiency.strategies.base import DataSelectionStrategy
from data_efficiency.strategies.k_center import KCenterGreedyStrategy
from data_efficiency.utils.embeddings import compute_entropy, get_predictions


class EDFSLiteStrategy(DataSelectionStrategy):
    """
    EDFS-lite strategy: Easy-and-Diverse-First.

    Splits data into bins by difficulty (easy, medium, hard) and applies
    k-center greedy within each bin.
    """

    def __init__(
        self,
        easy_portion: float = 0.4,
        medium_portion: float = 0.4,
        hard_portion: float = 0.2,
        model_name: str = "answerdotai/ModernBERT-base",
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        """
        Initialize EDFS-lite strategy.

        Args:
            easy_portion: Portion of budget for easy samples (default: 0.4)
            medium_portion: Portion of budget for medium samples (default: 0.4)
            hard_portion: Portion of budget for hard samples (default: 0.2)
            model_name: Model name for computing predictions
            batch_size: Batch size for processing
            num_workers: Number of workers for data loading
        """
        super().__init__()
        if abs(easy_portion + medium_portion + hard_portion - 1.0) > 1e-6:
            raise ValueError("Portions must sum to 1.0")
        self.easy_portion = easy_portion
        self.medium_portion = medium_portion
        self.hard_portion = hard_portion
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.k_center_strategy = KCenterGreedyStrategy(
            model_name=model_name, batch_size=batch_size, num_workers=num_workers
        )

    def _split_by_difficulty(
        self, dataset: TokenizedDataset, model, device: str
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Split dataset indices into easy, medium, and hard bins based on entropy.

        Args:
            dataset: Dataset to split
            model: Model for computing predictions
            device: Device to run computation on

        Returns:
            Tuple of (easy_indices, medium_indices, hard_indices)
        """
        # Compute predictions and entropy (on GPU)
        probs = get_predictions(
            model,
            dataset,
            device,
            model_name=self.model_name,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            return_numpy=False,  # Keep as torch tensor on GPU
        )
        entropy = compute_entropy(probs)  # Returns torch tensor

        # Split by quantiles: low entropy = easy, high entropy = hard
        n_samples = len(dataset)
        # Sort on GPU, then convert to numpy for indexing
        sorted_indices = torch.argsort(entropy).cpu().numpy()

        # Define quantile boundaries
        easy_boundary = int(n_samples * (1.0 / 3.0))
        medium_boundary = int(n_samples * (2.0 / 3.0))

        easy_indices = sorted_indices[:easy_boundary].tolist()
        medium_indices = sorted_indices[easy_boundary:medium_boundary].tolist()
        hard_indices = sorted_indices[medium_boundary:].tolist()

        return easy_indices, medium_indices, hard_indices

    def select(
        self, dataset: TokenizedDataset, limit: int, model=None, device: str = "cpu", **kwargs
    ) -> List[int]:
        """
        Select samples using EDFS-lite approach.

        Args:
            dataset: Dataset to select from
            limit: Number of samples to select
            model: Model for computing predictions and embeddings (required)
            device: Device to run computation on
            **kwargs: Additional arguments

        Returns:
            List of selected indices
        """
        if model is None:
            raise ValueError("Model is required for EDFS-lite strategy")

        if limit <= 0:
            return []

        n_samples = len(dataset)
        if limit >= n_samples:
            return list(range(n_samples))

        # Calculate budget for each bin
        budget_easy = int(limit * self.easy_portion)
        budget_medium = int(limit * self.medium_portion)
        budget_hard = limit - budget_easy - budget_medium  # Remaining goes to hard

        # Split dataset by difficulty
        easy_indices, medium_indices, hard_indices = self._split_by_difficulty(
            dataset, model, device
        )

        selected_indices = []

        # Select from each bin using k-center greedy
        if budget_easy > 0 and len(easy_indices) > 0:
            easy_dataset = dataset.select(easy_indices)
            easy_selected = self.k_center_strategy.select(
                easy_dataset,
                min(budget_easy, len(easy_indices)),
                model=model,
                device=device,
                **kwargs,
            )
            # Map back to original indices
            selected_indices.extend([easy_indices[idx] for idx in easy_selected])

        if budget_medium > 0 and len(medium_indices) > 0:
            medium_dataset = dataset.select(medium_indices)
            medium_selected = self.k_center_strategy.select(
                medium_dataset,
                min(budget_medium, len(medium_indices)),
                model=model,
                device=device,
                **kwargs,
            )
            # Map back to original indices
            selected_indices.extend([medium_indices[idx] for idx in medium_selected])

        if budget_hard > 0 and len(hard_indices) > 0:
            hard_dataset = dataset.select(hard_indices)
            hard_selected = self.k_center_strategy.select(
                hard_dataset,
                min(budget_hard, len(hard_indices)),
                model=model,
                device=device,
                **kwargs,
            )
            # Map back to original indices
            selected_indices.extend([hard_indices[idx] for idx in hard_selected])

        return selected_indices
