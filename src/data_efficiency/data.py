import random
from typing import Dict, Union

import torch
from datasets import Dataset as PyarrowDataset
from torch.utils.data import Dataset

from data_efficiency.utils.data import upload_dataset


class TokenizedDataset(Dataset):
    def __init__(self, dataset: PyarrowDataset):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, str]]:
        return self.dataset[index]

    def __len__(self) -> int:
        return len(self.dataset)


if __name__ == "__main__":
    dataset = TokenizedDataset(upload_dataset("test", data_dir="./data"))
    rand_idx = random.randint(0, len(dataset))
    rand_item = dataset[rand_idx]
    assert isinstance(rand_item, dict)
    assert all(
        [
            isinstance(value, torch.Tensor) or isinstance(value, str)
            for value in rand_item.values()
        ]
    )
    print(rand_item)
