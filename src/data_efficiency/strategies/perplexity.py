from transformers import AutoTokenizer, ModernBertForMaskedLM
import torch
from typing import List

from data_efficiency.data import TokenizedDataset
from data_efficiency.strategies.base import DataSelectionStrategy


class PerplexityDatasetSelectionStrategy(DataSelectionStrategy):
    """
    Selects examples by their per-token perplexity under a small language model:contentReference[oaicite:14]{index=14}.
    """

    def __init__(
        self,
        batch_size: int = 32,
        device: str = None,
        model_name: str = "answerdotai/ModernBERT-base",
    ):
        # Load a small pre-trained model (e.g. GPT2) for scoring
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = ModernBertForMaskedLM.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        self.model.to(self.device)
        self.model.eval()

    def select(self, dataset: TokenizedDataset, limit: int, **kwargs) -> List[int]:
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

                outputs = self.model(**inputs)
                logits = outputs.logits  # shape: [batch_size, seq_len, vocab_size]

                # Shift logits and labels for CrossEntropy
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs["input_ids"][..., 1:].contiguous()

                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                per_token_loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                ).view(shift_labels.size())

                # Average over tokens for each example
                batch_losses = per_token_loss.mean(dim=1).detach().cpu().tolist()
                losses.extend(batch_losses)

        # Lower loss → better perplexity
        indices = sorted(range(len(losses)), key=lambda i: losses[i])
        return indices[:limit]
