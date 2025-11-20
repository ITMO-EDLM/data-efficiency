import json
from typing import Callable, Dict, Optional

import click
from dotenv import load_dotenv

from .config import TrainingConfig
from .model import ModernBert
from .trainer import Trainer
from .utils import accuracy, f1, upload_dataset

# Mapping from metric names to functions
METRICS_MAP: Dict[str, Callable] = {
    "accuracy": accuracy,
    "f1": f1,
}


@click.command()
@click.option(
    "--config",
    "config_file",
    type=click.Path(exists=True),
    default=None,
    help="Path to JSON configuration file",
)
@click.option(
    "--device",
    "-d",
    type=str,
    default=None,
    help="Device to use (cpu, cuda, mps). Overrides config if provided.",
)
@click.option(
    "--run-name",
    "-r",
    type=str,
    default=None,
    help="Run name. Overrides config if provided.",
)
@click.option(
    "--use-clearml",
    is_flag=True,
    default=None,
    help="Enable ClearML logging. Overrides config if provided.",
)
def main(
    config_file: Optional[str],
    device: Optional[str],
    run_name: Optional[str],
    use_clearml: Optional[bool],
) -> None:
    """
    Train a model with data efficiency strategies.

    This tool will:
    - Load configuration from file or use defaults
    - Initialize model and datasets
    - Train model with selected data strategy
    - Log metrics to TensorBoard and optionally ClearML
    - Save checkpoints

    Examples:

        # From config file
        run --config configs/train_config_example.json

        # Override device from command line
        run --config configs/train_config_example.json -d cuda

        # Enable ClearML logging
        run --config configs/train_config_example.json --use-clearml
    """
    load_dotenv()
    # Load config from file if provided
    if config_file:
        with open(config_file, "r") as f:
            config_dict = json.load(f)
        config = TrainingConfig(**config_dict)
    else:
        # Use defaults
        config = TrainingConfig()

    # Override with command line arguments if provided
    if device is not None:
        config.device = device
    if run_name is not None:
        config.run_name = run_name
    if use_clearml is not None:
        config.use_clearml = use_clearml

    # Print configuration
    print("=" * 60)
    print("Training Configuration")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Device: {config.device}")
    print(f"Epochs: {config.n_epochs}")
    print(f"Strategy: {config.strategy_name}")
    print(f"Run name: {config.run_name}")
    print(f"ClearML: {config.use_clearml}")
    if config.use_clearml:
        print(f"ClearML Project: {config.clearml_project_name}")
        print(f"ClearML Task: {config.clearml_task_name or config.run_name}")
    print("=" * 60)

    # Initialize model
    model = ModernBert(
        backbone_name=config.model_name,
        num_classes=config.num_classes,
        dropout=config.dropout,
        unfreeze_layers=config.unfreeze_layers,
        use_pooler=config.use_pooler,
        use_float16=config.use_float16,
    )

    # Load datasets
    validation_dataset = upload_dataset("validation", data_dir=config.data_dir)
    train_dataset = upload_dataset("train", data_dir=config.data_dir)

    # Build metrics dictionary
    metrics_fn = {name: METRICS_MAP[name] for name in config.metrics if name in METRICS_MAP}
    if not metrics_fn:
        raise ValueError(f"No valid metrics found. Available: {list(METRICS_MAP.keys())}")

    # Convert config to dictionary for ClearML logging
    config_dict = config.model_dump()

    # Initialize trainer
    trainer = Trainer(
        model=model,
        loss_type=config.loss_type,
        metrics_fn=metrics_fn,
        val_dataset=validation_dataset,
        train_dataset=train_dataset,
        run_budget=config.run_budget,
        rounds_portions=config.rounds_portions,
        strategy_data={
            "strategy_name": config.strategy_name,
            "strategy_params": config.strategy_params,
        },
        optimizer_params=config.optimizer_params,
        n_epochs=config.n_epochs,
        device=config.device,
        model_name=config.model_name,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        log_dir=config.log_dir,
        run_name=config.run_name,
        checkpoint_dir=config.checkpoint_dir,
        save_checkpoints=config.save_checkpoints,
        max_checkpoints=config.max_checkpoints,
        checkpoint_metric=config.checkpoint_metric,
        use_clearml=config.use_clearml,
        clearml_project_name=config.clearml_project_name,
        clearml_task_name=config.clearml_task_name,
        config=config_dict,
    )

    # Run training
    trainer.setup()

    # Run hyperparameter tuning if enabled and strategy is "all"
    if config.enable_hyperparameter_tuning and config.strategy_name == "all":
        trainer.warmup()

    trainer.run()


if __name__ == "__main__":
    main()
