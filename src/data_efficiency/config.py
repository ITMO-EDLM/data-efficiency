from typing import Set

from pydantic import BaseModel


class GlobalConfig(BaseModel):
    dataset_path: str = "nyu-mll/glue"
    dataset_subset: str = "sst2"
    available_splits: Set[str] = {"train", "validation", "test"}
    data_dir: str = "./data"
    model_name: str = "answerdotai/ModernBERT-base"
    batch_size: int = 64
    num_workers: int = 4


global_config = GlobalConfig()
