"""Utility functions for computing lexical diversity metrics."""

from typing import List

import numpy as np


def compute_hdd(text: str) -> float:
    """
    Compute HD-D (Hypergeometric Distribution D) lexical diversity metric.

    HD-D measures vocabulary diversity by comparing observed word types
    to expected types under hypergeometric distribution.

    Args:
        text: Input text string

    Returns:
        HD-D score (higher = more diverse)
    """
    if not text or len(text.strip()) == 0:
        return 0.0

    tokens = text.lower().split()
    if len(tokens) == 0:
        return 0.0

    # Count word frequencies
    word_counts = {}
    for token in tokens:
        word_counts[token] = word_counts.get(token, 0) + 1

    n_types = len(word_counts)
    n_tokens = len(tokens)

    if n_tokens == 0:
        return 0.0

    # HD-D calculation: simplified version
    # More accurate version would use hypergeometric distribution
    # For now, use a proxy based on type-token ratio and sample size
    ttr = n_types / n_tokens
    # Adjust for sample size (larger samples tend to have lower TTR)
    hdd = ttr * np.sqrt(n_tokens)
    return hdd


def compute_mtld(text: str, threshold: float = 0.72) -> float:
    """
    Compute MTLD (Measure of Textual Lexical Diversity).

    MTLD measures the average number of words needed to reach a TTR threshold.

    Args:
        text: Input text string
        threshold: TTR threshold (default: 0.72)

    Returns:
        MTLD score (higher = more diverse)
    """
    if not text or len(text.strip()) == 0:
        return 0.0

    tokens = text.lower().split()
    if len(tokens) == 0:
        return 0.0

    n_tokens = len(tokens)
    if n_tokens < 2:
        return 0.0

    factors = []
    start_idx = 0

    while start_idx < n_tokens:
        types_seen = set()
        factor_length = 0

        for i in range(start_idx, n_tokens):
            types_seen.add(tokens[i])
            factor_length += 1
            current_ttr = len(types_seen) / factor_length

            if current_ttr < threshold:
                factors.append(factor_length)
                start_idx = i + 1
                break

        if start_idx < n_tokens and factor_length == n_tokens - start_idx:
            # Reached end without dropping below threshold
            factors.append(factor_length)
            break

    if len(factors) == 0:
        return 0.0

    mtld = n_tokens / np.mean(factors) if factors else 0.0
    return mtld


def compute_ttr(text: str) -> float:
    """
    Compute TTR (Type-Token Ratio) lexical diversity metric.

    TTR = number of unique words / total number of words

    Args:
        text: Input text string

    Returns:
        TTR score (0-1, higher = more diverse)
    """
    if not text or len(text.strip()) == 0:
        return 0.0

    tokens = text.lower().split()
    if len(tokens) == 0:
        return 0.0

    n_types = len(set(tokens))
    n_tokens = len(tokens)

    return n_types / n_tokens if n_tokens > 0 else 0.0


def compute_lexical_diversity(text: str, metric: str = "hdd") -> float:
    """
    Compute lexical diversity using specified metric.

    Args:
        text: Input text string
        metric: Metric to use ('hdd', 'mtld', or 'ttr')

    Returns:
        Lexical diversity score
    """
    if metric == "hdd":
        return compute_hdd(text)
    elif metric == "mtld":
        return compute_mtld(text)
    elif metric == "ttr":
        return compute_ttr(text)
    else:
        raise ValueError(f"Unknown metric: {metric}. Choose from 'hdd', 'mtld', 'ttr'")


