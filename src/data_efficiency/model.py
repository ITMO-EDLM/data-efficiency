from typing import List, Optional

from torch import nn, torch
from transformers import ModernBertConfig, ModernBertModel


class ModernBert(nn.Module):
    """
    Wrapper for loading backbone - ModernBERT model and add classifier layer
    with dropout.

    Layer freezing is controlled via the unfreeze_layers parameter:
    - None or 0: full backbone freezing (recommended for fine-tuning on small datasets)
    - 1-3: unfreeze last 1-3 layers (good for similar domains, small datasets)
    - 4-6: unfreeze last 4-6 layers (for medium domain differences)
    - 7-11: unfreeze most layers (for strongly different domains)
    - >= 12 (or >= total_layers): full backbone unfreezing (for large datasets)

    Recommendations for choosing the number of layers:
    1. Small datasets (< 1000 examples): 0-2 layers
    2. Medium datasets (1000-10000): 2-6 layers
    3. Large datasets (> 10000): 6-12 layers
    4. Similar domain: fewer layers (1-3)
    5. Different domain: more layers (6-12)
    """

    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        dropout: float,
        unfreeze_layers: Optional[int] = None,
        use_pooler: bool = False,
        use_float16: bool = False,
    ):
        super().__init__()
        self.config = ModernBertConfig.from_pretrained(backbone_name)
        self.backbone: ModernBertModel = ModernBertModel.from_pretrained(backbone_name)
        self.hidden = getattr(self.config, "hidden_dim", 768)

        self.use_pooler = use_pooler
        self.use_float16 = use_float16
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden, num_classes)

        # Layer freezing control via unfreeze_layers:
        # - None or 0: full backbone freezing
        # - N (1 <= N < total_layers): unfreeze last N encoder layers
        # - N >= total_layers: full backbone unfreezing
        if unfreeze_layers is None or unfreeze_layers == 0:
            # Full freezing
            for p in self.backbone.parameters():
                p.requires_grad = False
        else:
            # Determine total number of layers
            total_layers = None
            if hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layers"):
                total_layers = len(self.backbone.encoder.layers)

            if total_layers is not None and unfreeze_layers >= total_layers:
                # Full unfreezing - all backbone parameters are unfrozen
                for p in self.backbone.parameters():
                    p.requires_grad = True
            else:
                # Partial unfreezing: first freeze all, then unfreeze last N layers
                for p in self.backbone.parameters():
                    p.requires_grad = False

                if total_layers is not None:
                    layers_to_unfreeze = min(unfreeze_layers, total_layers)
                    # Unfreeze last layers_to_unfreeze encoder layers
                    for layer in self.backbone.encoder.layers[-layers_to_unfreeze:]:
                        for p in layer.parameters():
                            p.requires_grad = True
                else:
                    # Fallback: if structure is unexpected, unfreeze all
                    for p in self.backbone.parameters():
                        p.requires_grad = True

        # Convert to float16 if requested
        if use_float16:
            self.backbone = self.backbone.half()
            self.classifier = self.classifier.half()
            self.dropout = self.dropout.half()

    def get_head_params(self) -> List[torch.nn.Parameter]:
        """Returns head parameters (classifier + dropout)."""
        return list(self.classifier.parameters()) + list(self.dropout.parameters())

    def get_backbone_params(self) -> List[torch.nn.Parameter]:
        """Returns backbone parameters (only those that require gradients)."""
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        output = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        if (
            self.use_pooler
            and hasattr(output, "pooler_output")
            and output.pooler_output is not None
        ):
            embeddings = output.pooler_output
        else:
            embeddings = output.last_hidden_state[:, 0]  # CLS token

        x = self.dropout(embeddings)
        logits = self.classifier(x)
        return logits


if __name__ == "__main__":
    # Simple load tests
    model_name = "answerdotai/ModernBERT-base"
    # Test with full freezing
    model = ModernBert(model_name, 2, 0.2, unfreeze_layers=None, use_float16=True)
    assert model.config.hidden_size == model.hidden
    assert all([p.dtype == torch.float16 for p in model.classifier.parameters()])
    assert all([p.dtype == torch.float16 for p in model.backbone.parameters()])
    assert all([p.requires_grad is False for p in model.backbone.parameters()])

    # Test with full unfreezing (12 layers for ModernBERT-base)
    model_unfrozen = ModernBert(model_name, 2, 0.2, unfreeze_layers=12, use_float16=False)
    assert any([p.requires_grad is True for p in model_unfrozen.backbone.parameters()])
