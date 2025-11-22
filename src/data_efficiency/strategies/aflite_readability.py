from transformers import AutoTokenizer, ModernBertModel
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from typing import List

from data_efficiency.data import TokenizedDataset
from data_efficiency.strategies.base import DataSelectionStrategy


class AFLiteReadabilityDatasetSelectionStrategy(DataSelectionStrategy):
    """
    Selects examples by combining AFLite predictability (trained on real embeddings)
    and readability metrics.

    AFLite Implementation:
    1. Extracts embeddings using ModernBERT.
    2. repeatedly trains a linear classifier on random data splits.
    3. 'Predictability' is the ratio of times an example was correctly classified
       in the validation set.
    """

    def __init__(
        self,
        batch_size: int = 32,
        device: str = None,
        model_name: str = "answerdotai/ModernBERT-base",
        n_runs: int = 5,
        train_size: float = 0.8,
        seed: int = 42,
    ):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = ModernBertModel.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        # AFLite specific parameters
        self.n_runs = n_runs
        self.train_size = train_size
        self.seed = seed

        self.model.to(self.device)
        self.model.eval()

    def select(self, dataset: TokenizedDataset, limit: int, **kwargs) -> List[int]:
        n = len(dataset)

        # --- 1. Readability Score ---
        # Compute simple metric (negative avg word length)
        texts = [dataset[i].get("sentence", "") for i in range(n)]
        readability_scores = []
        for text in texts:
            words = text.split()
            avg_len = sum(len(w) for w in words) / max(len(words), 1)
            readability_scores.append(-avg_len)

        # --- 2. AFLite Predictability ---

        # A. Extract Embeddings
        # We need numerical representations (X) and targets (y) for the classifier
        labels = [dataset[i]["label"] for i in range(n)]
        all_embeddings = []

        with torch.no_grad():
            input_texts = [dataset[i].get("sentence", "") for i in range(n)]

            for start in range(0, n, self.batch_size):
                batch_texts = input_texts[start : start + self.batch_size]
                inputs = self.tokenizer(
                    batch_texts, return_tensors="pt", padding=True, truncation=True
                ).to(self.device)

                outputs = self.model(**inputs)
                # Use CLS token (index 0) or mean pooling as feature
                embeddings = outputs.last_hidden_state[:, 0, :]
                all_embeddings.append(embeddings.cpu().numpy())

        X = np.concatenate(all_embeddings, axis=0)
        y = np.array(labels)

        # B. Iterative Training (AFLite Loop)
        # We count how many times each example was in the val set
        # and how many times it was correctly predicted.
        correct_counts = np.zeros(n)
        total_counts = np.zeros(n)

        rs = ShuffleSplit(
            n_splits=self.n_runs, train_size=self.train_size, random_state=self.seed
        )

        for train_idx, val_idx in rs.split(X):
            # Train a lightweight model
            clf = LogisticRegression(max_iter=500, solver="lbfgs")
            clf.fit(X[train_idx], y[train_idx])

            # Predict on validation set
            preds = clf.predict(X[val_idx])

            # Update stats
            is_correct = preds == y[val_idx]
            correct_counts[val_idx] += is_correct.astype(int)
            total_counts[val_idx] += 1

        # Compute predictability score
        # Handle division by zero for examples that might never appear in validation (unlikely with enough runs)
        afl_scores = np.divide(
            correct_counts,
            total_counts,
            out=np.zeros_like(correct_counts, dtype=float),
            where=total_counts != 0,
        )

        # --- 3. Combine signals and select top examples ---
        rd = np.array(readability_scores)
        af = np.array(afl_scores)

        # Normalize scores to [0,1]
        rd_norm = (rd - rd.min()) / (rd.max() - rd.min() + 1e-12)
        af_norm = (af - af.min()) / (af.max() - af.min() + 1e-12)

        # Equal weighting
        combined = 0.5 * rd_norm + 0.5 * af_norm

        # Sort by combined score (higher is better)
        selected = list(np.argsort(combined)[-limit:])

        # Reverse to have the highest quality first
        selected.reverse()

        return selected
