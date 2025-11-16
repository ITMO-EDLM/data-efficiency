from typing import Any, Dict, Optional

from torch.optim import AdamW

from data_efficiency.model import ModernBert


def create_optimizer_with_different_lr(
    model: ModernBert,
    optimizer_params: Dict[str, Any],
    lr_head: Optional[float] = None,
    lr_backbone: Optional[float] = None,
) -> AdamW:
    """
    Creates an optimizer with different learning rates for head and backbone.

    Args:
        model: ModernBert model
        optimizer_params: Base optimizer parameters (weight_decay, betas, etc.)
        lr_head: Learning rate for head (if None, uses lr from optimizer_params)
        lr_backbone: Learning rate for backbone (if None, uses lr from optimizer_params)

    Returns:
        AdamW optimizer with parameter groups for head and backbone
    """
    # Base lr from optimizer_params
    base_lr = optimizer_params.get("lr", 2e-5)

    # Determine lr for head and backbone
    head_lr = lr_head if lr_head is not None else base_lr
    backbone_lr = lr_backbone if lr_backbone is not None else base_lr

    # Get head and backbone parameters
    head_params = model.get_head_params()
    backbone_params = model.get_backbone_params()

    # Create parameter groups
    param_groups = []

    # Group for head
    if head_params:
        head_group = {
            "params": head_params,
            "lr": head_lr,
        }
        # Copy remaining optimizer parameters (weight_decay, betas, etc.)
        for key, value in optimizer_params.items():
            if key != "lr":
                head_group[key] = value
        param_groups.append(head_group)

    # Group for backbone
    if backbone_params:
        backbone_group = {
            "params": backbone_params,
            "lr": backbone_lr,
        }
        # Copy remaining optimizer parameters
        for key, value in optimizer_params.items():
            if key != "lr":
                backbone_group[key] = value
        param_groups.append(backbone_group)

    # If no parameters with requires_grad=True, use all parameters with base lr
    if not param_groups:
        param_groups = [{"params": list(model.parameters()), **optimizer_params}]

    return AdamW(param_groups)
