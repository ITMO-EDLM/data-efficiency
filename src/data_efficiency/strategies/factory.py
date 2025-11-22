from typing import Any, Dict

from data_efficiency.strategies.all import AllDatasetSelectionStrategy
from data_efficiency.strategies.base import DataSelectionStrategy
from data_efficiency.strategies.edfs_lite import EDFSLiteStrategy
from data_efficiency.strategies.entropy_diversity import EntropyDiversityStrategy
from data_efficiency.strategies.k_center import KCenterGreedyStrategy
from data_efficiency.strategies.lexical_diversity import LexicalDiversityStrategy
from data_efficiency.strategies.qdit_lite import QDITLiteStrategy
from data_efficiency.strategies.random import RandomDatasetSelectionStrategy
from data_efficiency.strategies.perplexity import PerplexityDatasetSelectionStrategy
from data_efficiency.strategies.aflite_readability import (
    AFLiteReadabilityDatasetSelectionStrategy,
)
from data_efficiency.strategies.el2n import EL2NDatasetSelectionStrategy
from data_efficiency.strategies.ifd import IFDDatasetSelectionStrategy


def get_strategy(
    strategy_name: str, strategy_params: Dict[str, Any]
) -> DataSelectionStrategy:
    if strategy_name == "all":
        return AllDatasetSelectionStrategy(**strategy_params)
    elif strategy_name == "random":
        return RandomDatasetSelectionStrategy(**strategy_params)
    elif strategy_name == "perplexity":
        return PerplexityDatasetSelectionStrategy(**strategy_params)
    elif strategy_name == "aflite_readability":
        return AFLiteReadabilityDatasetSelectionStrategy(**strategy_params)
    elif strategy_name == "el2n":
        return EL2NDatasetSelectionStrategy(**strategy_params)
    elif strategy_name == "ifd":
        return IFDDatasetSelectionStrategy(**strategy_params)
    elif strategy_name == "k_center":
        return KCenterGreedyStrategy(**strategy_params)
    elif strategy_name == "entropy_diversity":
        return EntropyDiversityStrategy(**strategy_params)
    elif strategy_name == "lexical_diversity":
        return LexicalDiversityStrategy(**strategy_params)
    elif strategy_name == "qdit_lite":
        return QDITLiteStrategy(**strategy_params)
    elif strategy_name == "edfs_lite":
        return EDFSLiteStrategy(**strategy_params)
    else:
        raise ValueError(f"The {strategy_name} is not supported strategy type")
