import numpy as np


def cosine_distance_to_many(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """
    Computes cosine distance from one query vector to many candidate vectors.
    Inputs:
    - query (D,)
    - candidates (N, D)

    Output: distances (N,)

    Zero candidates are treated as distance +inf.
    Possible error: ValueError if query is zero vector.
    """
    if query.ndim != 1:
        raise ValueError("Query must be 1D")

    if candidates.ndim != 2:
        raise ValueError("Candidates must be 2D")

    if query.shape[0] != candidates.shape[1]:
        raise ValueError("Query and candidates must have matching dimensions")

    query = query.astype(np.float32)
    candidates = candidates.astype(np.float32)

    query_norm = np.float32(np.linalg.norm(query))
    if query_norm == 0:
        raise ValueError("Query vector is zero.")

    candidates_norm = np.linalg.norm(candidates, axis=1).astype(np.float32)
    zero_mask = candidates_norm == 0

    with np.errstate(divide="ignore", invalid="ignore"):  # handle division by zero
        sims = np.dot(candidates, query) / (candidates_norm * query_norm)

    one = np.float32(1.0)
    minus_one = np.float32(-1.0)

    sims = np.clip(sims, minus_one, one)
    sims[zero_mask] = 0.0  # avoid NaNs

    distances = one - sims

    inf = np.float32(np.inf)

    distances[zero_mask] = inf  # infinit distance for zero candidates
    return distances.astype(np.float32)


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
