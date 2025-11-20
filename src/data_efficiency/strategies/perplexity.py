from transformers import AutoTokenizer, ModernBertForMaskedLM
import torch
from typing import List

from data_efficiency.data import TokenizedDataset
from data_efficiency.strategies.base import DataSelectionStrategy


class PerplexityDatasetSelectionStrategy(DataSelectionStrategy):
    """
    Selects examples by their per-token perplexity under a small language model:contentReference[oaicite:14]{index=14}.
    """

    def __init__(self, batch_size: int = 32, device: str = None):
        # Load a small pre-trained model (e.g. GPT2) for scoring
        self.tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        self.model = ModernBertForMaskedLM.from_pretrained(
            "answerdotai/ModernBERT-base"
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        self.model.to(self.device)
        self.model.eval()

    def select(self, dataset: TokenizedDataset, limit: int) -> List[int]:
        texts = [dataset[i]["sentence"] for i in range(len(dataset))]

        losses = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch_texts = texts[start : start + self.batch_size]

                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.device)

                outputs = self.model(**inputs, labels=inputs["input_ids"])
                batch_losses = outputs.loss.detach().cpu().tolist()
                losses.extend(batch_losses)

        # Lower loss → better perplexity
        indices = sorted(range(len(losses)), key=lambda i: losses[i])
        return indices[:limit]