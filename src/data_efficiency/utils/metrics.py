import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def f1(probs: np.ndarray, preds: np.ndarray, gt: np.ndarray) -> float:
    return f1_score(gt, preds, average="macro", zero_division=0)


def accuracy(probs: np.ndarray, preds: np.ndarray, gt: np.ndarray) -> float:
    return accuracy_score(gt, preds)
