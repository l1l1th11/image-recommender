from pathlib import Path

import numpy as np

from image_recommender.io.img_loader import load_rgb
from image_recommender.recommender.query_helpers import (
    SUPPORTED_FEATURES,
    extract_query_features,
    get_score_arr,
    load_canonical_ids,
)
from image_recommender.util.logs import get_logger

logger = get_logger(__name__)


def single_image_query(
    query_path: str | Path,
    run_dir: Path | str,
    k: int,
    feature_types: list[str] | None = None,
    weights: dict[str, float] | None = None,
) -> list[tuple[int, float]]:
    """
    Runs a single image query and returns the top k most similar results.

    Input:
        query_path: Path to query image
        run_dir: Directory containing feature folders
        k: Number of top results to return
        feature_types: Optional subset of feature types to process
        weights: Optional weights (must match keys, sum to appr. 1)

    Output:
        List of (image_id, score) pairs sorted ascending (best match first)

    Raises:
    - ValueError if no features or candidates are available

    Notes:
    - Loads image from path
    - Extracts query features
    - Computes per-candidate scores across available features
    - Ranks candidates by score (ascending)
    - Returns top-k (id, score) pairs
    """
    # load image
    img_rgb = load_rgb(path=query_path)

    # extract queries
    queries_by_feature = extract_query_features(img_rgb=img_rgb, feature_types=feature_types)

    # get scores
    score_arr = get_score_arr(
        run_dir=run_dir,
        queries_by_feature=queries_by_feature,
        feature_types=feature_types,
        weights=weights,
    )

    # discover available features in run_dir
    run_dir = Path(run_dir)
    feature_dirs = sorted(
        [
            feature_dir
            for feature_dir in run_dir.iterdir()
            if feature_dir.is_dir() and feature_dir.name in SUPPORTED_FEATURES
        ]
    )

    # guard against corrupted run_dir
    if not feature_dirs:
        raise ValueError("No feature directories found")

    # select one feature type
    feature_type = feature_dirs[0].name

    # get canonical ids from present feature type
    canonical_ids = load_canonical_ids(run_dir=run_dir, feature_type=feature_type)

    # rank via indices (lowest -> highest)
    ranked_idxs = np.argsort(score_arr)

    # clamp k
    final_k = min(k, score_arr.shape[0])

    if final_k < k:
        logger.warning(
            f"Not enough scores to return top {k} neighbors, clamping to {final_k} neighbors"
        )

    # build top k output
    top_k = []

    for idx in ranked_idxs[:final_k]:
        top_k.append((canonical_ids[idx], score_arr[idx]))

    return top_k
