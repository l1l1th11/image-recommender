import numpy as np


def validate_input(distances: dict[str, np.ndarray]) -> int:
    # check min one feature was provided
    if not distances:
        raise ValueError("No distances were provided, can't calculate scores")

    # ensure all arrays have same length
    first_value = next(iter(distances.values()))
    n_candidates = len(first_value)

    for arr in distances.values():
        if len(arr) != n_candidates:
            raise ValueError("All distance arrays should have same length")

    return n_candidates
