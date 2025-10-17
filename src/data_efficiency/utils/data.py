from pathlib import Path
from typing import Any, Callable, Dict

import torch
from datasets import Dataset, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer, DataCollatorWithPadding, PreTrainedTokenizer

from data_efficiency.config import global_config


def download_dataset() -> None:
    dataset = load_dataset(path=global_config.dataset_path, name=global_config.dataset_subset)
    tokenizer = AutoTokenizer.from_pretrained(global_config.model_name)
    dataset = dataset.map(lambda x: {"label": 0 if x["label"] == -1 else x["label"]})

    def tokenize_batch(batch):
        return tokenizer(batch["sentence"], padding=False, truncation=True)

    tokenized_dataset = dataset.map(tokenize_batch, batched=True, desc="tokenizing")
    for split in global_config.available_splits:
        tokenized_split = tokenized_dataset[split]
        tokenized_split.set_format(
            type="torch", columns=["sentence", "input_ids", "attention_mask", "label", "idx"]
        )
        tokenized_split.save_to_disk(Path(global_config.data_dir, split))


def upload_dataset(split: str) -> Dataset:
    return load_from_disk(Path(global_config.data_dir, split))


def build_collate_fn(tokenizer: PreTrainedTokenizer) -> Callable:
    base = DataCollatorWithPadding(tokenizer=tokenizer)

    def collate_fn(features: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        batch_lbl = [f["label"] for f in features]
        model_feats = []
        for f in features:
            d = {
                k: f[k] for k in f.keys() if k in ("input_ids", "attention_mask", "token_type_ids")
            }
            model_feats.append(d)

        padded = base(model_feats)
        padded["labels"] = torch.tensor(batch_lbl, dtype=torch.long)
        return padded

    return collate_fn


def build_dataloader(dataset: TorchDataset, shuffle: bool = False) -> DataLoader:
    tokenizer = AutoTokenizer.from_pretrained(global_config.model_name)
    dloader = DataLoader(
        dataset=dataset,
        batch_size=global_config.batch_size,
        shuffle=shuffle,
        num_workers=global_config.num_workers,
        collate_fn=build_collate_fn(tokenizer),
    )
    return dloader


if __name__ == "__main__":
    download_dataset()
