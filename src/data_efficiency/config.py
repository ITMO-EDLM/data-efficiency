from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class TrainingConfig(BaseModel):
    """Configuration for model training."""

    # Model settings
    model_name: str = "answerdotai/ModernBERT-base"
    num_classes: int = 2
    dropout: float = 0.2
    unfreeze_layers: Optional[int] = None  # Number of layers to unfreeze:
    #   - None or 0: full backbone freezing (recommended for fine-tuning on small datasets)
    #   - 1-3: unfreeze last 1-3 layers (good for similar domains)
    #   - 4-6: unfreeze last 4-6 layers (for medium domain differences)
    #   - 7-12: unfreeze most/all layers (for strongly different domains or large datasets)
    #   - >= total_layers (usually 12 for ModernBERT-base): full backbone unfreezing
    use_pooler: bool = False
    use_float16: bool = False

    # Dataset settings
    dataset_path: str = "nyu-mll/glue"
    dataset_subset: str = "sst2"
    data_dir: str = "./data"

    # DataLoader settings
    batch_size: int = 64
    num_workers: int = 4

    # Training settings
    loss_type: str = "ce"
    n_epochs: int = 3
    device: str = "mps"
    run_budget: float = 1.0
    rounds_portions: List[float] = [0.5, 0.5]

    # Strategy settings
    strategy_name: str = "random"  # "random" or "all"
    strategy_params: Dict = {}

    # Optimizer settings
    # If separate lr is used, lr_head and lr_backbone will override lr
    optimizer_params: Dict = {"lr": 2e-5}
    lr_head: Optional[float] = None  # Learning rate for head (classifier + dropout)
    lr_backbone: Optional[float] = None  # Learning rate for backbone (if unfrozen)

    metrics: List[str] = ["accuracy", "f1"]

    # Logging and checkpoints
    log_dir: str = "./runs"
    run_name: Optional[str] = None
    checkpoint_dir: str = "./checkpoints"
    save_checkpoints: bool = True
    max_checkpoints: int = 3  # Maximum number of best checkpoints to keep
    checkpoint_metric: str = "val_loss"  # Metric to use for checkpoint selection

    # ClearML settings
    use_clearml: bool = False
    clearml_project_name: Optional[str] = None
    clearml_task_name: Optional[str] = None

    # Hyperparameter tuning settings
    enable_hyperparameter_tuning: bool = False  # Enable hyperparameter search
    warmup_epochs: int = 2  # Number of epochs for each search iteration
    tuning_n_iterations: int = 25  # Number of random combinations for Random Search
    tuning_sample_size: float = (
        0.15  # Fraction of train dataset for search (0.15 = 15%)
    )
    tuning_metric: str = (
        "val_loss"  # Metric for selecting best parameters ("val_loss" or "val_accuracy")
    )

    # Ranges for Random Search (optional, defaults available)
    batch_size_search_range: Optional[List[int]] = (
        None  # [min, max] or None for auto-detection
    )
    dropout_range: List[float] = Field([0.1, 0.5])
    lr_range: List[float] = Field(
        [1e-5, 1e-4]
    )  # Used if lr_head_range and lr_backbone_range are not specified
    lr_head_range: Optional[List[float]] = None  # Learning rate range for head
    lr_backbone_range: Optional[List[float]] = None  # Learning rate range for backbone
    weight_decay_options: Optional[List[float]] = Field([0.0, 0.01, 0.1])
    betas_options: Optional[List[List[float]]] = Field(
        [[0.9, 0.999], [0.95, 0.999], [0.9, 0.99]]
    )
    unfreeze_layers_options: Optional[List[int]] = (
        None  # Options for number of unfrozen layers to search
    )

    @model_validator(mode="before")
    @classmethod
    def handle_freeze_backbone_compatibility(cls, data):
        """
        Backward compatibility: converts freeze_backbone to unfreeze_layers.
        If freeze_backbone is present in config, convert it:
        - freeze_backbone=True -> unfreeze_layers=None (full freezing)
        - freeze_backbone=False -> unfreeze_layers=12 (full unfreezing, assume 12 layers)
        """
        if isinstance(data, dict):
            if "freeze_backbone" in data and "unfreeze_layers" not in data:
                freeze_backbone = data.pop("freeze_backbone")
                if freeze_backbone:
                    data["unfreeze_layers"] = None
                else:
                    # Full unfreezing - use large number (>= 12 for ModernBERT-base)
                    data["unfreeze_layers"] = 12
        return data


class EvaluationConfig(BaseModel):
    """Configuration for model evaluation."""

    # Model and checkpoint settings
    checkpoint_path: str
    model_name: str = "answerdotai/ModernBERT-base"
    num_classes: int = 2
    dropout: float = 0.2

    # Run configuration
    run_name: Optional[str] = None
    device: str = "mps"

    # Dataset splits to evaluate
    eval_splits: List[str] = ["validation", "test"]
    data_dir: str = "./data"

    # DataLoader settings
    batch_size: int = 64
    num_workers: int = 4

    # Metrics to compute
    compute_confusion_matrix: bool = True
    compute_auc_roc: bool = True
    compute_pr_curve: bool = True

    # Output paths
    artifacts_dir: str = "./artifacts"
    save_predictions: bool = False
    save_metrics_csv: bool = True
