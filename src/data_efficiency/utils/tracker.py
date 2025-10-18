from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


class MetricTracker:
    """
    Class for tracking run and round metrics
    """

    def __init__(
        self,
        metrics_fn: Dict[str, Callable],
        log_dir: Optional[str] = "./runs",
        run_name: Optional[str] = None,
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

    def save_train_loss(self, loss: torch.Tensor) -> None:
        loss_value = loss.cpu().item()
        self.train_loss_history[self.current_epoch].append(loss_value)

        # Log to TensorBoard
        if self.writer:
            self.writer.add_scalar("train/loss", loss_value, self.train_step)

    def save_vall_loss(self, loss: torch.Tensor) -> None:
        loss_value = loss.cpu().item()
        self.val_loss_history[self.current_epoch].append(loss_value)

        # Log to TensorBoard
        if self.writer:
            self.writer.add_scalar("val/loss", loss_value, self.val_step)

    def save_train_metrics(self, probs: np.ndarray, preds: np.ndarray, gt: np.ndarray) -> None:
        for metric_name, metric_fn in self.metrics_fn.items():
            metric_value = metric_fn(probs, preds, gt)
            self.train_metrics_history[self.current_epoch][metric_name].append(metric_value)

            # Log to TensorBoard
            if self.writer:
                self.writer.add_scalar(f"train/{metric_name}", metric_value, self.train_step)

        # Increment step counter after all metrics are logged
        self.train_step += 1

    def save_val_metrics(self, probs: np.ndarray, preds: np.ndarray, gt: np.ndarray) -> None:
        for metric_name, metric_fn in self.metrics_fn.items():
            metric_value = metric_fn(probs, preds, gt)
            self.val_metrics_history[self.current_epoch][metric_name].append(metric_value)

            # Log to TensorBoard
            if self.writer:
                self.writer.add_scalar(f"val/{metric_name}", metric_value, self.val_step)

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

    def close(self) -> None:
        """Close TensorBoard writer"""
        if self.writer:
            self.writer.close()
            print("TensorBoard writer closed")
