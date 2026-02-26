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
