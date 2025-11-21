import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import click
import numpy as np
import torch
import tqdm

from data_efficiency.config import EvaluationConfig
from data_efficiency.data import TokenizedDataset
from data_efficiency.model import ModernBert
from data_efficiency.utils import (
    build_dataloader,
    compute_confusion_matrix_data,
    compute_metrics,
    compute_pr_curve_data,
    compute_roc_curve_data,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
    save_metrics_to_csv,
    save_predictions_to_csv,
    upload_dataset,
)


class Evaluator:
    """Class for evaluating trained models on validation and test sets."""

    def __init__(self, config: EvaluationConfig):
        """
        Initialize evaluator with configuration.

        Args:
            config: EvaluationConfig instance
        """
        self.config = config
        self.device = torch.device(config.device)
        self.model: Optional[ModernBert] = None

        # Create artifacts directory
        self.artifacts_dir = Path(config.artifacts_dir)
        if config.run_name:
            self.artifacts_dir = self.artifacts_dir / config.run_name
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        print(f"Artifacts will be saved to: {self.artifacts_dir}")

    def load_checkpoint(self) -> None:
        """Load model from checkpoint."""
        checkpoint_path = Path(self.config.checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        # Initialize model (freeze backbone for inference)
        self.model = ModernBert(
            backbone_name=self.config.model_name,
            num_classes=self.config.num_classes,
            dropout=self.config.dropout,
            unfreeze_layers=None,  # Full freezing for inference
            use_pooler=False,
            use_float16=False,
        )

        # Load state dict
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded successfully (epoch {checkpoint.get('epoch', 'unknown')})")

    def evaluate_split(
        self, split_name: str
    ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate model on a specific dataset split.

        Args:
            split_name: Name of the split ('validation', 'test', etc.)

        Returns:
            Tuple of (metrics_dict, y_true, y_pred, y_probs)
        """
        print(f"\nEvaluating on {split_name} set...")

        # Load dataset
        dataset = TokenizedDataset(
            upload_dataset(split_name, data_dir=self.config.data_dir)
        )
        dataloader = build_dataloader(
            dataset,
            model_name=self.config.model_name,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
        )

        # Collect predictions
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader, desc=f"Evaluating {split_name}"):
                labels = batch.pop("labels")
                inputs = batch

                # Move to device
                for k, v in inputs.items():
                    inputs[k] = v.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                logits = self.model(**inputs)
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=-1)

                # Collect results
                all_preds.append(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        # Concatenate all batches
        y_pred = np.concatenate(all_preds)
        y_probs = np.concatenate(all_probs)
        y_true = np.concatenate(all_labels)

        # Compute metrics
        metrics = compute_metrics(y_true, y_pred, y_probs)

        print(f"\nResults on {split_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")

        return metrics, y_true, y_pred, y_probs

    def plot_visualizations(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: np.ndarray,
        split_name: str,
    ) -> None:
        """
        Generate and save all visualization plots.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_probs: Prediction probabilities
            split_name: Name of the split
        """
        split_dir = self.artifacts_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        # Confusion Matrix
        if self.config.compute_confusion_matrix:
            cm = compute_confusion_matrix_data(y_true, y_pred)
            plot_confusion_matrix(
                cm,
                class_names=["Negative", "Positive"],
                save_path=split_dir / "confusion_matrix.png",
                normalize=False,
            )
            # Also save normalized version
            plot_confusion_matrix(
                cm,
                class_names=["Negative", "Positive"],
                save_path=split_dir / "confusion_matrix_normalized.png",
                normalize=True,
            )

        # ROC Curve (for binary classification)
        if self.config.compute_auc_roc and y_probs.shape[1] == 2:
            fpr, tpr, auc_score = compute_roc_curve_data(y_true, y_probs[:, 1])
            plot_roc_curve(fpr, tpr, auc_score, save_path=split_dir / "roc_curve.png")

        # PR Curve (for binary classification)
        if self.config.compute_pr_curve and y_probs.shape[1] == 2:
            precision, recall, ap_score = compute_pr_curve_data(y_true, y_probs[:, 1])
            plot_pr_curve(
                precision, recall, ap_score, save_path=split_dir / "pr_curve.png"
            )

        # Save predictions if requested
        if self.config.save_predictions:
            save_predictions_to_csv(
                y_true,
                y_pred,
                y_probs,
                save_path=split_dir / "predictions.csv",
            )

    def run(self) -> None:
        """
        Run full evaluation pipeline.

        This will:
        1. Load the model checkpoint
        2. Evaluate on all configured splits
        3. Generate visualizations
        4. Save metrics to CSV
        """
        # Load model
        self.load_checkpoint()

        # Collect all metrics
        all_metrics = []

        # Evaluate each split
        for split_name in self.config.eval_splits:
            try:
                metrics, y_true, y_pred, y_probs = self.evaluate_split(split_name)

                # Generate visualizations
                self.plot_visualizations(y_true, y_pred, y_probs, split_name)

                # Store metrics with split name
                metric_row = {"split": split_name, **metrics}
                all_metrics.append(metric_row)

            except Exception as e:
                print(f"Error evaluating {split_name}: {e}")
                continue

        # Save metrics CSV
        if self.config.save_metrics_csv and all_metrics:
            metrics_path = self.artifacts_dir / "metrics.csv"
            save_metrics_to_csv(all_metrics, metrics_path)

        print(f"\n✓ Evaluation complete! Results saved to {self.artifacts_dir}")


@click.command()
@click.option(
    "--checkpoint",
    "-c",
    "checkpoint_path",
    type=click.Path(exists=True),
    required=False,
    default=None,
    help="Path to model checkpoint file",
)
@click.option(
    "--run-name",
    "-r",
    type=str,
    default=None,
    help="Run name for organizing artifacts (defaults to checkpoint directory name)",
)
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
    default="mps",
    help="Device to use (cpu, cuda, mps)",
)
@click.option(
    "--splits",
    "-s",
    type=str,
    default="validation,test",
    help="Comma-separated list of splits to evaluate",
)
@click.option(
    "--artifacts-dir",
    "-a",
    type=click.Path(),
    default="./artifacts",
    help="Directory to save artifacts",
)
@click.option(
    "--save-predictions",
    is_flag=True,
    default=False,
    help="Save prediction results to CSV",
)
def main(
    checkpoint_path: Optional[str],
    run_name: Optional[str],
    config_file: Optional[str],
    device: str,
    splits: str,
    artifacts_dir: str,
    save_predictions: bool,
) -> None:
    """
    Evaluate a trained model on validation and test sets.

    This tool will:
    - Load a trained model from checkpoint
    - Compute accuracy and F1 score
    - Generate confusion matrices, ROC and PR curves
    - Save all results and visualizations to the artifacts directory

    Examples:

        # Basic evaluation
        evaluate -c checkpoints/my_run/best/model.pt

        # With custom run name and device
        evaluate -c checkpoints/my_run/best/model.pt -r my_evaluation -d cuda

        # From config file
        evaluate --config eval_config.json
    """
    # Load config from file if provided
    if config_file:
        with open(config_file, "r") as f:
            config_dict = json.load(f)
        config = EvaluationConfig(**config_dict)
    else:
        # Require checkpoint_path if no config file provided
        if checkpoint_path is None:
            raise click.BadParameter(
                "Either --checkpoint or --config must be provided",
                param_hint="--checkpoint or --config",
            )
        # Infer run name from checkpoint path if not provided
        if run_name is None:
            checkpoint_path_obj = Path(checkpoint_path)
            # Try to get the run name from the checkpoint path structure
            # e.g., checkpoints/my_run/best/model.pt -> my_run
            if len(checkpoint_path_obj.parts) >= 3:
                run_name = checkpoint_path_obj.parts[-3]
            else:
                run_name = "evaluation"

        # Parse splits
        eval_splits = [s.strip() for s in splits.split(",")]

        # Create config from command line arguments
        config = EvaluationConfig(
            checkpoint_path=checkpoint_path,
            run_name=run_name,
            device=device,
            eval_splits=eval_splits,
            artifacts_dir=artifacts_dir,
            save_predictions=save_predictions,
        )

    # Print configuration
    print("=" * 60)
    print("Evaluation Configuration")
    print("=" * 60)
    print(f"Checkpoint: {config.checkpoint_path}")
    print(f"Run name: {config.run_name}")
    print(f"Device: {config.device}")
    print(f"Splits: {', '.join(config.eval_splits)}")
    print(f"Artifacts directory: {config.artifacts_dir}")
    print("=" * 60)

    # Create evaluator and run
    evaluator = Evaluator(config)
    evaluator.run()


if __name__ == "__main__":
    main()
