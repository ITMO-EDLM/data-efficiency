from transformers import AutoTokenizer, ModernBertForMaskedLM
import torch
from typing import List, Dict
import numpy as np

from data_efficiency.data import TokenizedDataset
from data_efficiency.strategies.base import DataSelectionStrategy


class IFDDatasetSelectionStrategy(DataSelectionStrategy):
    """
    Adapts IFD (Instruction-Following Difficulty) for Classification tasks (like SST-2).

    It measures the influence of the input 'sentence' on the 'label'.
    IFD = Loss(Label | No Context) - Loss(Label | Sentence)

    High IFD means the sentence is very informative for the label.
    """

    def __init__(
        self,
        batch_size: int = 32,
        device: str = None,
        model_name: str = "answerdotai/ModernBERT-base",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = ModernBertForMaskedLM.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        # SST-2 specific mapping
        self.label_map = {0: "negative", 1: "positive"}

        self.model.to(self.device)
        self.model.eval()

    def _compute_loss(self, text_list: List[str]) -> List[float]:
        """
        Computes the generative loss (perplexity-style) for the given texts.
        Using ModernBERT with shifted logits to simulate autoregressive scoring.
        """
        losses = []
        with torch.no_grad():
            for start in range(0, len(text_list), self.batch_size):
                batch_texts = text_list[start : start + self.batch_size]

                inputs = self.tokenizer(
                    batch_texts, return_tensors="pt", padding=True, truncation=True
                ).to(self.device)

                outputs = self.model(**inputs)
                logits = outputs.logits

                # Shift logits and labels for CrossEntropy (Causal language modeling style)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs["input_ids"][..., 1:].contiguous()

                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                per_token_loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                ).view(shift_labels.size())

                # Mean loss per example
                batch_losses = per_token_loss.mean(dim=1).detach().cpu().tolist()
                losses.extend(batch_losses)
        return losses

    def select(self, dataset: TokenizedDataset, limit: int) -> List[int]:
        n = len(dataset)

        # 1. Prepare Data
        # We need to format the classification task as text generation.
        texts_with_context = []
        texts_without_context = []

        for i in range(n):
            sentence = dataset[i]["sentence"]
            label_id = dataset[i]["label"]
            label_text = self.label_map.get(label_id, str(label_id))

            # Condition A: Predict label GIVEN the sentence
            # Format: "Review: <sentence> Sentiment: <label>"
            full_text = f"Review: {sentence} Sentiment: {label_text}"
            texts_with_context.append(full_text)

            # Condition B: Predict label GIVEN NOTHING (or just the template)
            # Format: "Sentiment: <label>"
            # This measures the prior probability of the label
            blind_text = f"Sentiment: {label_text}"
            texts_without_context.append(blind_text)

        # 2. Compute Losses
        loss_with = self._compute_loss(texts_with_context)
        loss_without = self._compute_loss(texts_without_context)

        # 3. Calculate IFD
        ifd_scores = []
        for lw, lwo in zip(loss_with, loss_without):
            # IFD = Loss_without_Context - Loss_with_Context
            # If IFD is High: The sentence helped reduce the loss significantly (Good Data)
            # If IFD is Low/Negative: The sentence confused the model or didn't help (Noisy/Hard Data)
            score = lwo - lw
            ifd_scores.append(score)

        # 4. Select
        # Usually for IFD, we want the examples with the *Highest* IFD score
        # (where the input instruction/sentence was most useful).
        # Note: The original code selected 'lowest' IFD because they treated it as "Difficulty".
        # However, in data filtering, we usually want High Information Gain.

        # Let's stick to the standard interpretation:
        # We keep examples where the sentence strongly implies the label.
        # Sort descending (Highest IFD first)
        indices = np.argsort(ifd_scores)[::-1][:limit]

        return list(indices)
