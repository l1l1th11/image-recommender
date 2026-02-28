import numpy as np


def chi_distance_to_many(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """
    Computes chi-squared distances from one query histogram to many candidate histograms.

    Input: Histogram-like vectors.

    - Self-distance: If query = candidate, then χ² = 0.
    - Symmetry: (query - candidate)^2 / (query + candidate) = (candidate - query)^2 / (candidate + query)
    - No negativity: χ² >= 0

    Explanation:

    A large difference in one bin with small query + candidate --> large impact.
    A large difference in one bin with large query + candidate --> smaller impact.
    """
    # Condition:
    query = query.astype(np.float32)
    candidates = candidates.astype(np.float32)

    if query.shape[0] != candidates.shape[1]:
        raise ValueError("Query and candidates must have matching number of bins")

    eps = 1e-10  # to avoid division by zero

    # vectorized chi-squared computation
    num = (candidates - query) ** 2
    denom = candidates + query + eps
    chi_sq = 0.5 * np.sum(num / denom, axis=1)

    return chi_sq


# Scalar version for one candidate:


def chi_distance(query: np.ndarray, candidate: np.ndarray) -> float:
    """
    Computes the chi-squared distance between one query histogram and a single candidate histogram.

    Input: Histogram-like vectors.

    - Self-distance: If query = candidate, then χ² = 0.
    - Symmetry: χ²(query, candidate) = χ²(candidate, query)
    - No negativity: χ² >= 0

    This is a scalar wrapper around chi_distance_to_many.
    """
    return chi_distance_to_many(query, candidate.reshape(1, -1))[0]
