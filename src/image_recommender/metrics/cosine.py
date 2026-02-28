import numpy as np


def cosine_distance_to_many(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """
    Computes cosine distance from one query vector to many candidate vectors.
    Input: query (D,), candidates (N, D)
    Output: distances (N,)
    Zero candidates are treated as distance +inf.
    Possible error: ValueError if query is zero vector.
    """
    query = query.astype(np.float32)
    candidates = candidates.astype(np.float32)

    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        raise ValueError("Query vector is zero.")

    candidates_norm = np.linalg.norm(candidates, axis=1)
    zero_mask = candidates_norm == 0

    with np.errstate(divide="ignore", invalid="ignore"):  # handle division by zero
        sims = np.dot(candidates, query) / (candidates_norm * query_norm)

    sims = np.clip(sims, -1.0, 1.0)
    sims[zero_mask] = 0.0  # avoid NaNs

    distances = 1.0 - sims
    distances[zero_mask] = np.inf  # infinit distance for zero candidates
    return distances


# Scalar version for one candidate:


def cosine_distance(query: np.ndarray, candidate: np.ndarray) -> float:
    """
    Computes cosine distance from one query vector to one candidate vector.
    Input: query (D,), candidate (D,)
    Output: distance (scalar)
    Zero candidates are treated as distance +inf.
    Possible error: ValueError if query is zero vector.
    """
    return cosine_distance_to_many(query, candidate.reshape(1, -1))[0]
