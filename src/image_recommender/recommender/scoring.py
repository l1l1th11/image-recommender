import math
import numbers

import numpy as np

from image_recommender.config import VAR_EPSILON


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
    epsilon = VAR_EPSILON

    for feature, dist_arr in norm_dist_dict.items():
        # only keep features with meaningful variation (range >= epsilon)
        var_range = np.max(dist_arr) - np.min(dist_arr)
        if var_range >= epsilon:
            filtered_dist_dict[feature] = dist_arr

    return filtered_dist_dict


def score_candidates(filtered_dist_dict: dict[str, np.ndarray]) -> np.ndarray:
    # stack distance arrays
    dist_arrs = list(filtered_dist_dict.values())
    stacked_dist_arrs = np.stack(dist_arrs, axis=0)

    # compute mean along feature axis
    score_arr = np.mean(stacked_dist_arrs, axis=0)

    return score_arr.astype(np.float32)


def compute_scores(
    dist_dict: dict[str, np.ndarray], weights: dict[str, float] | None = None
) -> np.ndarray:
    """
    Compute a final score per candidate by combining multiple feature distances.

    Steps:
    - validate input alignment
    - normalize distances to [0, 1]
    - drop near constant features (range < VAR_EPSILON)
    - fuse remaining features via mean or weighted sum (weights renormalized after filtering)

    Args:
        dist_dict: per feature distance arrays (same length & order)
        weights: optional weights (must match keys, sum to appr. 1)

    Returns:
        float32 score array (aligned with input, lower = better)

    Edge case:
        no active features -> zero array
    """
    # validate input and get number of candidates
    n_candidates = validate_input(dist_dict=dist_dict)

    # normalize dict
    norm_dist_dict = normalize_dict(dist_dict=dist_dict)

    # filter 0 variance
    filtered_dist_dict = filter_0_variance(norm_dist_dict=norm_dist_dict)

    # handle empty distance dicts
    if not filtered_dist_dict:
        return np.zeros(shape=(n_candidates,), dtype=np.float32)

    # compute scores
    if weights is None:
        score_arr = score_candidates(filtered_dist_dict=filtered_dist_dict)

    else:
        # ensure provided features match input
        if not set(weights.keys()) == set(dist_dict.keys()):
            raise ValueError("Given weights features don't align with provided data")

        # check provided weights sum to 1
        if not math.isclose(sum(weights.values()), 1, rel_tol=1e-6):
            raise ValueError("Given weights don't sum to 1")

        # check weights are numeric & >= 0
        for weight in weights.values():
            if not isinstance(weight, numbers.Number):
                raise ValueError("Given weights must be numeric")
            if weight < 0:
                raise ValueError("Given weights must be >= 0")

        # select active weights
        active_weights = {}
        for feature in filtered_dist_dict:
            active_weights[feature] = weights[feature]

        # renormalize (divide by sum of active weights)
        renormalized_weights = {}
        sum_act_weights = sum(active_weights.values())

        for feature, weight in active_weights.items():
            renormalized_weights[feature] = weight / sum_act_weights

        # stack distance arrays
        dist_arrs = list(filtered_dist_dict.values())
        stacked_dist_arrs = np.stack(dist_arrs, axis=0)

        # get ordered weight list
        ordered_weights_list = []

        for feature in filtered_dist_dict:
            ordered_weights_list.append(renormalized_weights[feature])

        # convert to array
        ordered_weights = np.array(ordered_weights_list)

        # reshape
        reshaped_weights = ordered_weights[:, None]

        # apply weights per row
        weighted_dist_arrs = stacked_dist_arrs * reshaped_weights

        # compute sum feature axis
        score_arr = np.sum(weighted_dist_arrs, axis=0)

    return score_arr.astype(np.float32)
