from transformers import AutoTokenizer, ModernBertModel
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import List

from data_efficiency.data import TokenizedDataset
from data_efficiency.strategies.base import DataSelectionStrategy


class EL2NDatasetSelectionStrategy(DataSelectionStrategy):
    """
    Selects examples by the L2 norm of prediction error under a proxy classifier.
    Uses ModernBERT to extract features, then trains a lightweight classifier.
    """

    def __init__(
        self,
        batch_size: int = 32,
        device: str = None,
        model_name: str = "answerdotai/ModernBERT-base",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = ModernBertModel.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        self.model.to(self.device)
        self.model.eval()

    def select(self, dataset: TokenizedDataset, limit: int) -> List[int]:
        texts = [dataset[i]["sentence"] for i in range(len(dataset))]
        labels = [dataset[i]["label"] for i in range(len(dataset))]

        all_embeddings = []

        # 1. Extract Embeddings using ModernBERT
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch_texts = texts[start : start + self.batch_size]
                inputs = self.tokenizer(
                    batch_texts, return_tensors="pt", padding=True, truncation=True
                ).to(self.device)

                outputs = self.model(**inputs)
                # Use the first token (CLS equivalent) or mean pool as the feature vector
                # ModernBERT output shape: [batch, seq_len, hidden_dim]
                batch_embeddings = outputs.last_hidden_state[:, 0, :]
                all_embeddings.append(batch_embeddings.cpu().numpy())

        X = np.concatenate(all_embeddings, axis=0)
        y = np.array(labels)

        # 2. Train Proxy Classifier (Logistic Regression)
        # We increase max_iter to ensure convergence on high-dim BERT embeddings
        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf.fit(X, y)
        probs = clf.predict_proba(X)  # shape (N, num_classes)

        # 3. Compute EL2N
        num_classes = probs.shape[1]
        # Create one-hot encoding of true labels
        one_hots = np.eye(num_classes)[y]

        # Calculate L2 norm of the difference vector
        el2n_scores = np.linalg.norm(probs - one_hots, axis=1)

        # Select examples with smallest error norm (easiest/cleanest data)
        # Argsort is ascending, so [:limit] gives lowest scores
        selected = list(np.argsort(el2n_scores)[:limit])
        return selected
