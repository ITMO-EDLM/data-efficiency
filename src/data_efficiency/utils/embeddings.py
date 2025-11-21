"""Utility functions for computing embeddings and predictions from models."""

from typing import Union

import numpy as np
import torch
import tqdm

from data_efficiency.data import TokenizedDataset
from data_efficiency.model import ModernBert
from data_efficiency.utils.data import build_dataloader

# Configure TensorFloat32 (TF32) for better performance on Ampere+ GPUs
# This enables faster float32 matrix multiplications without significant precision loss
if torch.cuda.is_available():
    # Use new API for PyTorch 2.9+
    try:
        torch.set_float32_matmul_precision("high")  # Enables TF32 for matmul
    except AttributeError:
        # Fallback for older PyTorch versions
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True


def get_embeddings(
    model: ModernBert,
    dataset: TokenizedDataset,
    device: str,
    model_name: str = "answerdotai/ModernBERT-base",
    batch_size: int = 64,
    num_workers: int = 4,
    return_numpy: bool = False,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Compute embeddings for all samples in the dataset.

    Args:
        model: The model to use for computing embeddings
        dataset: Dataset to compute embeddings for
        device: Device to run computation on
        model_name: Model name for tokenizer
        batch_size: Batch size for processing
        num_workers: Number of workers for data loading
        return_numpy: If True, return numpy array (CPU). If False, return torch tensor (on device).

    Returns:
        Array/tensor of shape (n_samples, embedding_dim) with embeddings
    """
    model.eval()
    dataloader = build_dataloader(
        dataset,
        model_name=model_name,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    embeddings_list = []
    with torch.no_grad():
        for batch in dataloader:
            _ = batch.pop("labels")  # Remove labels, not used for embeddings
            inputs = batch
            for k, v in inputs.items():
                inputs[k] = v.to(device)

            # Get embeddings from backbone
            output = model.backbone(**inputs, return_dict=True)
            if (
                model.use_pooler
                and hasattr(output, "pooler_output")
                and output.pooler_output is not None
            ):
                embeddings = output.pooler_output
            else:
                embeddings = output.last_hidden_state[:, 0]  # CLS token

            if return_numpy:
                embeddings_list.append(embeddings.cpu().numpy())
            else:
                embeddings_list.append(embeddings)

    if return_numpy:
        return np.concatenate(embeddings_list, axis=0)
    else:
        return torch.cat(embeddings_list, dim=0)


def get_predictions(
    model: ModernBert,
    dataset: TokenizedDataset,
    device: str,
    model_name: str = "answerdotai/ModernBERT-base",
    batch_size: int = 64,
    num_workers: int = 4,
    return_numpy: bool = False,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Compute prediction probabilities for all samples in the dataset.

    Args:
        model: The model to use for predictions
        dataset: Dataset to compute predictions for
        device: Device to run computation on
        model_name: Model name for tokenizer
        batch_size: Batch size for processing
        num_workers: Number of workers for data loading
        return_numpy: If True, return numpy array (CPU). If False, return torch tensor (on device).

    Returns:
        Array/tensor of shape (n_samples, n_classes) with prediction probabilities
    """
    model.eval()
    dataloader = build_dataloader(
        dataset,
        model_name=model_name,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    probs_list = []
    with torch.no_grad():
        print("Start calculating predictions")
        for batch in tqdm.tqdm(dataloader):
            _ = batch.pop("labels")  # Remove labels, not used for predictions
            inputs = batch
            for k, v in inputs.items():
                inputs[k] = v.to(device)

            logits = model(**inputs)
            probs = torch.softmax(logits, dim=1)
            if return_numpy:
                probs_list.append(probs.cpu().numpy())
            else:
                probs_list.append(probs)

    if return_numpy:
        return np.concatenate(probs_list, axis=0)
    else:
        return torch.cat(probs_list, dim=0)


def compute_entropy(probs: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute entropy for each sample's prediction distribution.

    Args:
        probs: Array/tensor of shape (n_samples, n_classes) with probabilities

    Returns:
        Array/tensor of shape (n_samples,) with entropy values
    """
    # Avoid log(0) by adding small epsilon
    eps = 1e-10

    if isinstance(probs, torch.Tensor):
        # GPU-optimized version using torch
        probs_clipped = torch.clamp(probs, min=eps, max=1.0 - eps)
        entropy = -torch.sum(probs_clipped * torch.log(probs_clipped), dim=1)
        return entropy
    else:
        # CPU version using numpy (backward compatibility)
        probs_clipped = np.clip(probs, eps, 1.0 - eps)
        entropy = -np.sum(probs_clipped * np.log(probs_clipped), axis=1)
        return entropy


def compute_distances_gpu(
    embeddings: torch.Tensor, selected: torch.Tensor
) -> torch.Tensor:
    """
    Compute pairwise distances between all embeddings and selected embeddings on GPU.

    Args:
        embeddings: Tensor of shape (n_samples, embedding_dim) on GPU
        selected: Tensor of shape (n_selected, embedding_dim) on GPU

    Returns:
        Tensor of shape (n_samples, n_selected) with pairwise distances
    """
    # Use torch.cdist for efficient batch distance computation
    # p=2 means Euclidean distance
    distances = torch.cdist(embeddings, selected, p=2)
    return distances


def compute_min_distances_gpu(
    embeddings: torch.Tensor, selected: torch.Tensor
) -> torch.Tensor:
    """
    Compute minimum distance from each embedding to any selected embedding on GPU.

    Args:
        embeddings: Tensor of shape (n_samples, embedding_dim) on GPU
        selected: Tensor of shape (n_selected, embedding_dim) on GPU

    Returns:
        Tensor of shape (n_samples,) with minimum distances
    """
    distances = compute_distances_gpu(embeddings, selected)
    min_distances = torch.min(distances, dim=1)[0]
    return min_distances
