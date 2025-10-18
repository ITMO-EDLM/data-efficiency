from torch import nn, torch
from transformers import ModernBertConfig, ModernBertModel

from data_efficiency.config import global_config


class ModernBert(nn.Module):
    """
    Wrapper for loading backbone - ModernBERT model and add classifier layer
    with dropout
    """

    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        dropout: float,
        freeze_backbone: bool = True,
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

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Convert to float16 if requested
        if use_float16:
            self.backbone = self.backbone.half()
            self.classifier = self.classifier.half()
            self.dropout = self.dropout.half()

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
    model = ModernBert(global_config.model_name, 2, 0.2, True, False, True)
    assert model.config.hidden_size == model.hidden
    assert all([p.dtype == torch.float16 for p in model.classifier.parameters()])
    assert all([p.dtype == torch.float16 for p in model.backbone.parameters()])
    assert all([p.requires_grad is False for p in model.backbone.parameters()])
