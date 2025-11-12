from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

try:
    from clearml import Task
except ImportError:
    Task = None


class MetricTracker:
    """
    Class for tracking run and round metrics.
    Supports both TensorBoard and ClearML logging.
    """

    def __init__(
        self,
        metrics_fn: Dict[str, Callable],
        log_dir: Optional[str] = "./runs",
        run_name: Optional[str] = None,
        use_clearml: bool = False,
        clearml_project_name: Optional[str] = None,
        clearml_task_name: Optional[str] = None,
    ):
        self.metrics_fn = metrics_fn
        self.current_epoch = 1
        self.train_loss_history: Dict[int, List[float]] = {self.current_epoch: []}
        self.val_loss_history: Dict[int, List[float]] = {self.current_epoch: []}
        self.train_metrics_history: Dict[int, Dict[str, List[float]]] = {
            self.current_epoch: {metric_name: [] for metric_name in self.metrics_fn}
        }
        self.val_metrics_history: Dict[int, Dict[str, List[float]]] = {
            self.current_epoch: {metric_name: [] for metric_name in self.metrics_fn}
        }

        # TensorBoard logging
        self.train_step = 0
        self.val_step = 0

        if log_dir:
            # Create run-specific directory
            if run_name:
                full_log_dir = str(Path(log_dir) / run_name)
            else:
                # Use timestamp as default run name
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                full_log_dir = str(Path(log_dir) / timestamp)

            self.writer = SummaryWriter(log_dir=full_log_dir)
            print(f"TensorBoard logs will be saved to: {full_log_dir}")
            print(f"Run 'tensorboard --logdir={log_dir}' to view")
        else:
            self.writer = None

        # ClearML logging
        self.clearml_task = None
        if use_clearml:
            if Task is None:
                print("Warning: ClearML is not installed. Install it with: pip install clearml")
            else:
                try:
                    # ClearML reads credentials from environment variables automatically
                    # CLEARML_API_ACCESS_KEY and CLEARML_API_SECRET_KEY
                    project_name = clearml_project_name or "DataEfficiency"
                    task_name = (
                        clearml_task_name or run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
                    )

                    self.clearml_task = Task.init(project_name=project_name, task_name=task_name)
                    print(f"ClearML task initialized: {project_name}/{task_name}")
                except Exception as e:
                    print(f"Warning: Failed to initialize ClearML task: {e}")
                    print(
                        "Make sure CLEARML_API_ACCESS_KEY and "
                        "CLEARML_API_SECRET_KEY are set in environment"
                    )
                    self.clearml_task = None

    def save_train_loss(self, loss: torch.Tensor) -> None:
        loss_value = loss.cpu().item()
        self.train_loss_history[self.current_epoch].append(loss_value)

        # Log to TensorBoard
        if self.writer:
            self.writer.add_scalar("train/loss", loss_value, self.train_step)

        # Log to ClearML
        if self.clearml_task:
            self.clearml_task.get_logger().report_scalar(
                title="train", series="loss", value=loss_value, iteration=self.train_step
            )

    def save_vall_loss(self, loss: torch.Tensor) -> None:
        loss_value = loss.cpu().item()
        self.val_loss_history[self.current_epoch].append(loss_value)

        # Log to TensorBoard
        if self.writer:
            self.writer.add_scalar("val/loss", loss_value, self.val_step)

        # Log to ClearML
        if self.clearml_task:
            self.clearml_task.get_logger().report_scalar(
                title="val", series="loss", value=loss_value, iteration=self.val_step
            )

    def save_train_metrics(self, probs: np.ndarray, preds: np.ndarray, gt: np.ndarray) -> None:
        for metric_name, metric_fn in self.metrics_fn.items():
            metric_value = metric_fn(probs, preds, gt)
            self.train_metrics_history[self.current_epoch][metric_name].append(metric_value)

            # Log to TensorBoard
            if self.writer:
                self.writer.add_scalar(f"train/{metric_name}", metric_value, self.train_step)

            # Log to ClearML
            if self.clearml_task:
                self.clearml_task.get_logger().report_scalar(
                    title="train", series=metric_name, value=metric_value, iteration=self.train_step
                )

        # Increment step counter after all metrics are logged
        self.train_step += 1

    def save_val_metrics(self, probs: np.ndarray, preds: np.ndarray, gt: np.ndarray) -> None:
        for metric_name, metric_fn in self.metrics_fn.items():
            metric_value = metric_fn(probs, preds, gt)
            self.val_metrics_history[self.current_epoch][metric_name].append(metric_value)

            # Log to TensorBoard
            if self.writer:
                self.writer.add_scalar(f"val/{metric_name}", metric_value, self.val_step)

            # Log to ClearML
            if self.clearml_task:
                self.clearml_task.get_logger().report_scalar(
                    title="val", series=metric_name, value=metric_value, iteration=self.val_step
                )

        # Increment step counter after all metrics are logged
        self.val_step += 1

    def train_loss_info(self) -> None:
        avg_by_epochs = {k: np.mean(v) for k, v in self.train_loss_history.items()}
        avg_loss = np.mean(list(avg_by_epochs.values()))
        print(f"Average train loss: {avg_loss:4f}")
        print(f"Average train loss by last epoch: {avg_by_epochs[self.current_epoch]}")

    def val_loss_info(self) -> None:
        avg_by_epochs = {k: np.mean(v) for k, v in self.val_loss_history.items()}
        avg_loss = np.mean(list(avg_by_epochs.values()))
        print(f"Average val loss: {avg_loss:4f}")
        print(f"Average val loss by last epoch: {avg_by_epochs[self.current_epoch]}")

    def train_metrics_info(self) -> None:
        avg_by_epochs = defaultdict(dict)
        for epoch, metrics_history in self.train_metrics_history.items():
            for metric_name, metric_values in metrics_history.items():
                avg_by_epochs[metric_name][epoch] = np.mean(metric_values)

        for metric_name, avg_values in avg_by_epochs.items():
            print(f"Average train {metric_name}: {np.mean(list(avg_values.values())):2f}")
            print(f"Average train {metric_name} by last epoch: {list(avg_values.values())[-1]:2f}")

    def val_metrics_info(self) -> None:
        avg_by_epochs = defaultdict(dict)
        for epoch, metrics_history in self.val_metrics_history.items():
            for metric_name, metric_values in metrics_history.items():
                avg_by_epochs[metric_name][epoch] = np.mean(metric_values)

        for metric_name, avg_values in avg_by_epochs.items():
            print(f"Average val {metric_name}: {np.mean(list(avg_values.values())):2f}")
            print(f"Average val {metric_name} by last epoch: {list(avg_values.values())[-1]:2f}")

    def end_epoch(self) -> None:
        self.train_loss_info()
        self.val_loss_info()
        self.train_metrics_info()
        self.val_metrics_info()

        self.current_epoch += 1
        self.train_loss_history[self.current_epoch] = []
        self.val_loss_history[self.current_epoch] = []
        self.train_metrics_history[self.current_epoch] = {
            metric_name: [] for metric_name in self.metrics_fn
        }
        self.val_metrics_history[self.current_epoch] = {
            metric_name: [] for metric_name in self.metrics_fn
        }

    def log_configuration(self, config: Dict[str, Any]) -> None:
        """
        Log configuration to ClearML.

        Args:
            config: Dictionary with configuration parameters
        """
        if self.clearml_task:
            try:
                # Convert config to flat dictionary for ClearML
                # ClearML's connect() method accepts nested dictionaries
                self.clearml_task.connect(config)
                print("Configuration logged to ClearML")
            except Exception as e:
                print(f"Warning: Failed to log configuration to ClearML: {e}")

    def close(self) -> None:
        """Close TensorBoard writer and ClearML task"""
        if self.writer:
            self.writer.close()
            print("TensorBoard writer closed")

        if self.clearml_task:
            self.clearml_task.close()
            print("ClearML task closed")
