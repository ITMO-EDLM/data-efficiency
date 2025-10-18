from typing import Any, Dict

from data_efficiency.strategies.all import AllDatasetSelectionStrategy
from data_efficiency.strategies.base import DataSelectionStrategy
from data_efficiency.strategies.random import RandomDatasetSelectionStrategy


def get_strategy(strategy_name: str, strategy_params: Dict[str, Any]) -> DataSelectionStrategy:
    if strategy_name == "all":
        return AllDatasetSelectionStrategy(**strategy_params)
    elif strategy_name == "random":
        return RandomDatasetSelectionStrategy(**strategy_params)
    else:
        raise ValueError(f"The {strategy_name} is not supported strategy type")
