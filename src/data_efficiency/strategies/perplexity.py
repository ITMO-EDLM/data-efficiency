from transformers import AutoTokenizer, ModernBertForMaskedLM
import torch
from data_efficiency.data import TokenizedDataset
from data_efficiency.strategies.base import DataSelectionStrategy


class PerplexityStrategy(DataSelectionStrategy):
    """
    Selects examples by their per-token perplexity under a small language model:contentReference[oaicite:14]{index=14}.
    """

    def __init__(self):
        # Load a small pre-trained model (e.g. GPT2) for scoring
        self.tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        self.model = ModernBertForMaskedLM.from_pretrained(
            "answerdotai/ModernBERT-base"
        )
        self.model.eval()

    def select(self, dataset: TokenizedDataset, limit: int) -> List[int]:
        losses = []
        for i in range(len(dataset)):
            text = dataset[i][
                "response_text"
            ]  # model the response (or entire input+response)
            inputs = self.tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            losses.append(loss)

        # Lower loss -> lower perplexity. Select lowest-perplexity examples.
        indices = sorted(range(len(losses)), key=lambda i: losses[i])
        return indices[:limit]
