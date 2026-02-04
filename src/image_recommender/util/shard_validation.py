from collections.abc import Sequence

import numpy as np


def validate_shard(
    feature_type: str,
    features: np.ndarray,
    ids: Sequence[int],
    meta: dict[str, object],
    required_keys: set[str],
    expected_version: int,
) -> None:
    # check features dimensions
    if features.ndim != 2:
        raise ValueError("Feature array has wrong dimensions.")

    # check length matches
    if len(ids) != features.shape[0]:
        raise ValueError("The number of features and ids is mismatched.")

    # check meta keys
    if set(meta.keys()) != required_keys:
        raise ValueError("The meta keys are incorrect.")

    # check meta version
    if meta["version"] != expected_version:
        raise ValueError("Meta version mismatch.")

    # check consistency with data
    if meta["feature_dim"] != features.shape[1]:
        raise ValueError("Metadata and data feature dimensions are mismatched.")

    try:
        meta_dt = np.dtype(meta["feature_dtype"])
    except (TypeError, ValueError) as e:
        raise ValueError("Metadata type is invalid.") from e

    if meta_dt != features.dtype:
        raise ValueError("Metadata and data type are mismatched.")

    if meta["shard_size"] != features.shape[0]:
        raise ValueError("Metadata and data size are mismatched.")

    if meta["feature_type"] != feature_type:
        raise ValueError("Metadata and data feature type are mismatched.")
