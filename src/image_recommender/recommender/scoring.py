import numpy as np


def validate_input(dist_dict: dict[str, np.ndarray]) -> int:
    # check min one feature was provided
    if not dist_dict:
        raise ValueError("No dist_dict were provided, can't calculate scores")

    # ensure all arrays have same length
    first_value = next(iter(dist_dict.values()))
    n_candidates = len(first_value)

    for arr in dist_dict.values():
        if len(arr) != n_candidates:
            raise ValueError("All distance arrays should have same length")

    return n_candidates


def normalize_array(dist_arr: np.ndarray) -> np.ndarray:
    # compute min and max
    arr_min = np.min(dist_arr)
    arr_max = np.max(dist_arr)

    # if min == max, return 0 array of same shape
    if arr_min == arr_max:
        return np.zeros(shape=dist_arr.shape, dtype=np.float32)

    # min max normalization
    norm_dist_array = (dist_arr - arr_min) / (arr_max - arr_min)

    return norm_dist_array.astype(np.float32)


def normalize_dict(dist_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    norm_dist_dict = {}

    for feature, dist_arr in dist_dict.items():
        # normalize each array
        norm_dist_dict[feature] = normalize_array(dist_arr=dist_arr)

    return norm_dist_dict


def filter_0_variance(norm_dist_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    filtered_dist_dict = {}

    for feature, dist_arr in norm_dist_dict.items():
        # only keep arrays with variance
        if not np.all(dist_arr == dist_arr[0]):
            filtered_dist_dict[feature] = dist_arr

    return filtered_dist_dict


def score_candidates(filtered_dist_dict: dict[str, np.ndarray]) -> np.ndarray:
    # stack distance arrays
    dist_arrs = list(filtered_dist_dict.values())
    stacked_dist_arrs = np.stack(dist_arrs, axis=0)

    # compute mean along feature axis
    score_arr = np.mean(stacked_dist_arrs, axis=0)

    return score_arr.astype(np.float32)
