from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray
) -> Dict[str, float]:
    """
    Compute standard classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_probs: Prediction probabilities (for binary classification, shape (n, 2))

    Returns:
        Dictionary of metric names and values
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    # Add binary classification specific metrics
    if y_probs.shape[1] == 2:
        # Use probability of positive class
        pos_probs = y_probs[:, 1]

        # ROC-AUC
        fpr, tpr, _ = roc_curve(y_true, pos_probs)
        metrics["auc_roc"] = auc(fpr, tpr)

        # PR-AUC
        metrics["auc_pr"] = average_precision_score(y_true, pos_probs)

    return metrics


def compute_confusion_matrix_data(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Confusion matrix as numpy array
    """
    return confusion_matrix(y_true, y_pred)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    normalize: bool = False,
) -> None:
    """
    Plot confusion matrix as a heatmap.

    Args:
        cm: Confusion matrix
        class_names: List of class names for labels
        save_path: Path to save the plot
        normalize: Whether to normalize the confusion matrix
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        fmt = "d"

    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")
    plt.close()


def compute_roc_curve_data(
    y_true: np.ndarray, y_probs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute ROC curve data for binary classification.

    Args:
        y_true: Ground truth labels (0 or 1)
        y_probs: Prediction probabilities for positive class

    Returns:
        Tuple of (fpr, tpr, auc_score)
    """
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, auc_score


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot ROC curve.

    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc_score: AUC score
        save_path: Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"ROC curve saved to {save_path}")
    plt.close()


def compute_pr_curve_data(
    y_true: np.ndarray, y_probs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute Precision-Recall curve data for binary classification.

    Args:
        y_true: Ground truth labels (0 or 1)
        y_probs: Prediction probabilities for positive class

    Returns:
        Tuple of (precision, recall, average_precision)
    """
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    ap_score = average_precision_score(y_true, y_probs)
    return precision, recall, ap_score


def plot_pr_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    ap_score: float,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot Precision-Recall curve.

    Args:
        precision: Precision values
        recall: Recall values
        ap_score: Average precision score
        save_path: Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="darkorange", lw=2, label=f"PR curve (AP = {ap_score:.3f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"PR curve saved to {save_path}")
    plt.close()


def save_metrics_to_csv(
    metrics_data: List[Dict[str, any]],
    save_path: Path,
) -> None:
    """
    Save metrics to CSV file.

    Args:
        metrics_data: List of dictionaries containing metrics
        save_path: Path to save the CSV file
    """
    df = pd.DataFrame(metrics_data)
    df.to_csv(save_path, index=False)
    print(f"Metrics saved to {save_path}")


def save_predictions_to_csv(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    indices: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
) -> None:
    """
    Save predictions to CSV file for error analysis.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_probs: Prediction probabilities
        indices: Sample indices
        save_path: Path to save the CSV file
    """
    data = {
        "true_label": y_true,
        "predicted_label": y_pred,
        "is_correct": y_true == y_pred,
    }

    # Add probability columns
    for i in range(y_probs.shape[1]):
        data[f"prob_class_{i}"] = y_probs[:, i]

    if indices is not None:
        data["sample_idx"] = indices

    df = pd.DataFrame(data)

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Predictions saved to {save_path}")
