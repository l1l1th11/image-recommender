from pathlib import Path

import numpy as np

from image_recommender.recommender.single_image_query import _compute_full_scores
from image_recommender.util.logs import get_logger

logger = get_logger(__name__)


def multi_image_query(
    query_paths: list[str | Path],
    run_dir: Path | str,
    k: int,
    feature_types: list[str] | None = None,
    weights: dict[str, float] | None = None,
) -> tuple[list[tuple[int, float]], set[str]]:
    """
    Runs a multi image query by aggregating aligned per query score arrays via mean.

    Input:
        query_paths: List of query image paths
        run_dir: Directory containing feature folders
        k: Number of top results to return
        feature_types: Optional subset of feature types to process
        weights: Optional weights (must match keys, sum to appr. 1)

    Output:
        top_k: List of (image_id, score) pairs sorted ascending (best match first)
        used_features: Set of actually used feature types

    Raises:
        ValueError: If fewer than 2 query images are provided, or if query runs are inconsistent
    """
    # only use pipeline for multi image queries
    if len(query_paths) < 2:
        raise ValueError("Multi image query requires at least 2 query images")

    score_arrays = []
    reference_ids = None
    reference_features = None

    for query_path in query_paths:
        # get aligned score array, canonical id list and used features set
        score_arr, canonical_ids, used_features = _compute_full_scores(
            query_path=query_path,
            run_dir=run_dir,
            feature_types=feature_types,
            weights=weights,
        )

        # use first query as reference for canonical id ordering & used features
        if reference_ids is None:
            reference_ids = canonical_ids
            reference_features = used_features

        # check following queries for mismatch
        else:
            if canonical_ids != reference_ids:
                raise ValueError("Canonical candidate ids differ across query runs")

            if used_features != reference_features:
                raise ValueError("Used feature sets differ across query runs")

        # ensure score array shapes match
        if score_arrays and score_arr.shape != score_arrays[0].shape:
            raise ValueError("Score array shapes differ across query runs")

        score_arrays.append(score_arr)

    # stack per query score arrays, preserving alignment
    stacked_scores = np.stack(score_arrays, axis=0)  # (Q, N)

    # aggregate scores via mean
    aggregated_scores = np.mean(stacked_scores, axis=0)  # (N,)

    # rank via indices (lowest -> highest)
    ranked_idxs = np.argsort(aggregated_scores)

    # clamp k
    final_k = min(k, aggregated_scores.shape[0])

    if final_k < k:
        logger.warning(
            f"Not enough scores to return top {k} neighbors, clamping to {final_k} neighbors"
        )

    # build top k output
    top_k = []

    for idx in ranked_idxs[:final_k]:
        top_k.append((reference_ids[idx], aggregated_scores[idx]))

    # return validated feature set
    used_features = reference_features

    return top_k, used_features
