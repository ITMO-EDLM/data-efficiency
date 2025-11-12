from typing import Dict, List, Optional

from pydantic import BaseModel


class TrainingConfig(BaseModel):
    """Configuration for model training."""

    # Model settings
    model_name: str = "answerdotai/ModernBERT-base"
    num_classes: int = 2
    dropout: float = 0.2
    freeze_backbone: bool = True
    use_pooler: bool = False
    use_float16: bool = False

    # Dataset settings
    dataset_path: str = "nyu-mll/glue"
    dataset_subset: str = "sst2"
    train_size: Optional[int] = None  # If None, use full dataset
    val_size: Optional[int] = None  # If None, use full dataset
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
    optimizer_params: Dict = {"lr": 2e-5}

    # Metrics
    metrics: List[str] = ["accuracy", "f1"]

    # Logging and checkpoints
    log_dir: str = "./runs"
    run_name: Optional[str] = None
    checkpoint_dir: str = "./checkpoints"
    save_checkpoints: bool = True

    # ClearML settings
    use_clearml: bool = False
    clearml_project_name: Optional[str] = None
    clearml_task_name: Optional[str] = None


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
