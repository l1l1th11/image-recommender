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


def normalize(dist_arr: np.ndarray) -> np.ndarray:
    # compute min and max
    arr_min = np.min(dist_arr)
    arr_max = np.max(dist_arr)

    # if min == max, return 0 array of same shape
    if arr_min == arr_max:
        return np.zeros(shape=dist_arr.shape, dtype=np.float32)

    # min max normalization
    norm_dist_array = (dist_arr - arr_min) / (arr_max - arr_min)

    return norm_dist_array.astype(np.float32)
