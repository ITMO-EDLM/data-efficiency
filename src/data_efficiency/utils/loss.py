from torch import nn


def get_loss(loss_type: str) -> nn.Module:
    if loss_type == "bce":
        return nn.BCEWithLogitsLoss()
    elif loss_type == "ce" or loss_type == "cross_entropy":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Loss type '{loss_type}' not supported! Use 'bce' or 'ce'.")
