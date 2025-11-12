from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import torch
from datasets import Dataset, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer, DataCollatorWithPadding, PreTrainedTokenizer


def download_dataset(
    dataset_path: str = "nyu-mll/glue",
    dataset_subset: str = "sst2",
    model_name: str = "answerdotai/ModernBERT-base",
    data_dir: str = "./data",
    validation_size: int = 2000,
    random_seed: int = 42,
) -> None:
    """
    Download and prepare GLUE SST2 dataset with custom splits.

    In GLUE SST2, the test set has no labels. So we:
    1. Use original validation split as test (it has labels)
    2. Create validation split from train (~2k samples, stratified by labels)
    3. Keep remaining train samples as train

    Args:
        dataset_path: Path to the dataset on HuggingFace Hub
        dataset_subset: Subset name of the dataset
        model_name: Model name for tokenizer
        data_dir: Directory to save the processed dataset splits
        validation_size: Number of samples to select for validation (default: 2000)
        random_seed: Random seed for reproducibility (default: 42)
    """
    dataset = load_dataset(path=dataset_path, name=dataset_subset)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = dataset.map(lambda x: {"label": 0 if x["label"] == -1 else x["label"]})

    def tokenize_batch(batch):
        return tokenizer(batch["sentence"], padding=False, truncation=True)

    tokenized_dataset = dataset.map(tokenize_batch, batched=True, desc="tokenizing")

    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # 1. Original validation -> test (it has labels, unlike GLUE test)
    if "validation" in tokenized_dataset:
        test_split = tokenized_dataset["validation"]
        test_split.set_format(
            type="torch", columns=["sentence", "input_ids", "attention_mask", "label", "idx"]
        )
        test_split.save_to_disk(Path(data_dir, "test"))
        print(f"Saved test split: {len(test_split)} samples")

    # 2. Create validation split from train (stratified by labels)
    if "train" in tokenized_dataset:
        train_split = tokenized_dataset["train"]

        # Get labels for stratification
        labels = train_split["label"]
        unique_labels = np.unique(labels)

        # Calculate how many samples per class
        samples_per_class = validation_size // len(unique_labels)
        remainder = validation_size % len(unique_labels)

        validation_indices = []
        train_indices = []

        # Stratified sampling: select samples from each class proportionally
        for i, label in enumerate(unique_labels):
            label_indices = np.where(np.array(labels) == label)[0]
            # Shuffle indices for this label
            np.random.shuffle(label_indices)

            # Calculate how many samples to take for this class
            n_samples = samples_per_class + (1 if i < remainder else 0)
            n_samples = min(n_samples, len(label_indices))

            # Select validation samples
            val_idx = label_indices[:n_samples]
            validation_indices.extend(val_idx.tolist())

            # Remaining samples go to train
            train_idx = label_indices[n_samples:]
            train_indices.extend(train_idx.tolist())

        # Shuffle validation and train indices
        np.random.shuffle(validation_indices)
        np.random.shuffle(train_indices)

        # Create validation split
        validation_split = train_split.select(validation_indices)
        validation_split.set_format(
            type="torch", columns=["sentence", "input_ids", "attention_mask", "label", "idx"]
        )
        validation_split.save_to_disk(Path(data_dir, "validation"))
        print(f"Saved validation split: {len(validation_split)} samples")

        # Create remaining train split
        train_remaining = train_split.select(train_indices)
        train_remaining.set_format(
            type="torch", columns=["sentence", "input_ids", "attention_mask", "label", "idx"]
        )
        train_remaining.save_to_disk(Path(data_dir, "train"))
        print(f"Saved train split: {len(train_remaining)} samples")

        # Print label distribution
        print("\nLabel distribution:")
        val_labels = validation_split["label"]
        train_labels = train_remaining["label"]
        for label in unique_labels:
            val_count = sum(1 for label_val in val_labels if label_val == label)
            train_count = sum(1 for label_train in train_labels if label_train == label)
            print(f"  Label {label}: validation={val_count}, train={train_count}")


def upload_dataset(split: str, data_dir: str = "./data") -> Dataset:
    """
    Load a dataset split from disk.

    Args:
        split: Name of the split to load ('train', 'validation', 'test')
        data_dir: Directory where the dataset splits are stored

    Returns:
        Dataset loaded from disk
    """
    return load_from_disk(Path(data_dir, split))


class CustomCollator:
    """Collate function that can be pickled for multiprocessing."""

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.base = DataCollatorWithPadding(tokenizer=tokenizer)

    def __call__(self, features: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        batch_lbl = [f["label"] for f in features]
        model_feats = []
        for f in features:
            d = {
                k: f[k] for k in f.keys() if k in ("input_ids", "attention_mask", "token_type_ids")
            }
            model_feats.append(d)

        padded = self.base(model_feats)
        padded["labels"] = torch.tensor(batch_lbl, dtype=torch.long)
        return padded


def build_collate_fn(tokenizer: PreTrainedTokenizer) -> Callable:
    return CustomCollator(tokenizer)


def build_dataloader(
    dataset: TorchDataset,
    model_name: str = "answerdotai/ModernBERT-base",
    batch_size: int = 64,
    num_workers: int = 4,
    shuffle: bool = False,
) -> DataLoader:
    """
    Build a DataLoader for a dataset.

    Args:
        dataset: PyTorch dataset to load
        model_name: Model name for tokenizer
        batch_size: Batch size for the DataLoader
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle the data

    Returns:
        DataLoader instance
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=build_collate_fn(tokenizer),
    )
    return dloader


if __name__ == "__main__":
    download_dataset()
