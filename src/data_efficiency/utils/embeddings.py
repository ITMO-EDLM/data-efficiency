"""Utility functions for computing embeddings and predictions from models."""

from typing import Dict, List, Optional

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

from data_efficiency.data import TokenizedDataset
from data_efficiency.model import ModernBert
from data_efficiency.utils.data import build_dataloader


def get_embeddings(
    model: ModernBert,
    dataset: TokenizedDataset,
    device: str,
    model_name: str = "answerdotai/ModernBERT-base",
    batch_size: int = 64,
    num_workers: int = 4,
) -> np.ndarray:
    """
    Compute embeddings for all samples in the dataset.

    Args:
        model: The model to use for computing embeddings
        dataset: Dataset to compute embeddings for
        device: Device to run computation on
        model_name: Model name for tokenizer
        batch_size: Batch size for processing
        num_workers: Number of workers for data loading

    Returns:
        Array of shape (n_samples, embedding_dim) with embeddings
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
            labels = batch.pop("labels")
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

            embeddings_list.append(embeddings.cpu().numpy())

    return np.concatenate(embeddings_list, axis=0)


def get_predictions(
    model: ModernBert,
    dataset: TokenizedDataset,
    device: str,
    model_name: str = "answerdotai/ModernBERT-base",
    batch_size: int = 64,
    num_workers: int = 4,
) -> np.ndarray:
    """
    Compute prediction probabilities for all samples in the dataset.

    Args:
        model: The model to use for predictions
        dataset: Dataset to compute predictions for
        device: Device to run computation on
        model_name: Model name for tokenizer
        batch_size: Batch size for processing
        num_workers: Number of workers for data loading

    Returns:
        Array of shape (n_samples, n_classes) with prediction probabilities
    """
    model.eval()
    dataloader = build_dataloader(
        dataset.select(range(10)),
        model_name=model_name,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    probs_list = []
    with torch.no_grad():
        print("Start calculating predictions")
        for batch in tqdm.tqdm(dataloader):
            labels = batch.pop("labels")
            inputs = batch
            for k, v in inputs.items():
                inputs[k] = v.to(device)

            logits = model(**inputs)
            probs = torch.softmax(logits, dim=1)
            probs_list.append(probs.cpu().numpy())

    return np.concatenate(probs_list, axis=0)


def compute_entropy(probs: np.ndarray) -> np.ndarray:
    """
    Compute entropy for each sample's prediction distribution.

    Args:
        probs: Array of shape (n_samples, n_classes) with probabilities

    Returns:
        Array of shape (n_samples,) with entropy values
    """
    # Avoid log(0) by adding small epsilon
    eps = 1e-10
    probs_clipped = np.clip(probs, eps, 1.0 - eps)
    entropy = -np.sum(probs_clipped * np.log(probs_clipped), axis=1)
    return entropy
