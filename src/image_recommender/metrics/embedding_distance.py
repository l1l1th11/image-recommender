import numpy as np


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Computes cosine distance between two 1D vectors.
    Input: a, b - 1D numpy arrays of the same shape.
    Output: scalar float in [0, 2], where 0 means identical and 2 means opposite.
    Possible error: ValueError if shapes differ or if either vector is zero.
    """

    if a.ndim != 1 or b.ndim != 1:  # Are both inputs 1D vectors?
        raise ValueError("Both inputs must be 1D vectors.")

    if a.shape != b.shape:  # Do they have the same shape?
        raise ValueError("Vectors must have the same shape.")

    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0.0 or norm_b == 0.0:  # Is either vector a zero vector?
        raise ValueError("Cosine distance undefined for zero vectors.")

    sim = np.dot(a, b) / (norm_a * norm_b)

    sim = np.clip(sim, -1.0, 1.0)  # Clip the similarity value to avoid numerical issues

    return float(1.0 - sim)


def cosine_distance_to_many(q: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Computes cosine distance between a single query vector and
    multiple vectors in a matrix.
    Input: q (1D numpy array of shape (D,)), X (2D numpy array of shape (N, D)).
    Output: 1D numpy array of shape (N,) with cosine distances.
    Possible error: ValueError if shapes differ or if either vector is zero.
    """

    if q.ndim != 1:  # Is the query vector 1D?
        raise ValueError("Query vector must be 1D.")

    if X.ndim != 2:  # Is X a 2D matrix?
        raise ValueError("X must be a 2D array of shape (N, D).")

    if X.shape[1] != q.shape[0]:  # Does the dimensionality of q match the number of columns in X?
        raise ValueError("Dimensionality mismatch between q and X.")

    q = q.astype(np.float64, copy=False)
    X = X.astype(np.float64, copy=False)

    norm_q = np.linalg.norm(q)
    norms_X = np.linalg.norm(X, axis=1)

    if norm_q == 0.0:  # Is the query vector a zero vector?
        raise ValueError("Cosine distance undefined for zero query vector.")

    if np.any(norms_X == 0.0):  # Are there any zero vectors in X?
        raise ValueError("Cosine distance undefined for zero vector in X.")

    sims = (X @ q) / (norms_X * norm_q)

    sims = np.clip(sims, -1.0, 1.0)

    return 1.0 - sims  # Similarity ==> Distance


# 1 - 0 = 1 ==> Orthogonal
# 1 - 1 = 0 ==> Identical
# 1 - (-1) = 2 ==> Opposite
