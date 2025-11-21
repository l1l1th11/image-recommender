import numpy as np


def hsv_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Computes the chi-squared distance (relative divergence) between two histograms a and b.

    Input: Histogram-like vectors (ideally normalized, non-negative, same shape).

    - Self-distance: If a = b, then χ² = 0.
    - Symmetry: (a - b)^2 / (a + b) = (b - a)^2 / (b + a)
    - No negativity: χ² >= 0

    Explanation:

    A large difference in one bin with small a + b --> large impact.
    A large difference in one bin with large a + b --> smaller impact.
    """
    # Condition:
    if a.shape != b.shape:  # If the vectors a and b are not of the same shape...
        raise ValueError("Feature vectors must have the same shape")  # ... raise an error.

    # Define the vectors datatype
    a = a.astype(np.float64)
    b = b.astype(np.float64)

    eps: float = 1e-10  # to avoid division by zero
    chi: float = 0.5 * np.sum((a - b) ** 2 / (a + b + eps))

    return chi
