from pathlib import Path

import numpy as np

from image_recommender.io.img_loader import load_rgb
from image_recommender.recommender.query_helpers import (
    distances_all_features_subset,
    extract_query_features,
    get_score_arr,
)
from image_recommender.recommender.single_image_query import _compute_full_scores
from image_recommender.search.annoy import AnnoySearchBackend
from image_recommender.util.logs import get_logger

logger = get_logger(__name__)


def multi_image_query(
    query_paths: list[str | Path],
    run_dir: Path | str,
    k: int,
    feature_types: list[str] | None = None,
    weights: dict[str, float] | None = None,
    backend: str = "linear",
    k_candidates: int | None = None,
    annoy_backend: AnnoySearchBackend | None = None,
    id_to_vec_maps=None,
) -> tuple[list[tuple[int, float]], set[str]]:
    """
    Runs a multi image query by aggregating per query score arrays via mean.

    Input:
        query_paths: List of query image paths (must contain at least 2 images)
        run_dir: Directory containing feature folders
        k: Number of top results to return
        feature_types: Optional subset of feature types to process
        weights: Optional weights (must match keys, sum to approx. 1)
        backend: "linear" (full scan) or "annoy" (subset based search)
        k_candidates: Size of candidate subset retrieved by annoy (if applicable)
        annoy_backend: Pre-initialized annoy backend (recommended for repeated queries)
        id_to_vec_maps: Precomputed mappings {feature_type: {image_id: feature_vector}} required for annoy backend

    Output:
        top_k: List of (image_id, score) pairs sorted ascending (best match first)
        used_features: Set of actually used feature types

    Raises:
        ValueError: If fewer than 2 query images are provided, or if query runs are inconsistent

    Notes:
        - The first query runs the full pipeline and defines the candidate subset
        - Remaining queries reuse this subset and skip candidate generation
        - This avoids repeated expensive pipeline steps and significantly improves performance
        - All score arrays are aligned to the same canonical ID order before aggregation
    """

    if len(query_paths) < 2:
        raise ValueError("Multi image query requires at least 2 query images")

    # run full pipeline for first query to define candidate subset and reference alignment
    score_arr, canonical_ids, used_features, _ = _compute_full_scores(
        query_path=query_paths[0],
        run_dir=run_dir,
        feature_types=feature_types,
        weights=weights,
        backend=backend,
        k_candidates=k_candidates,
        annoy_backend=annoy_backend,
        id_to_vec_maps=id_to_vec_maps,
    )

    reference_ids = canonical_ids
    reference_features = used_features
    score_arrays = [score_arr]

    # process remaining queries on the fixed candidate subset
    for query_path in query_paths[1:]:

        # load image and extract features (no candidate generation here)
        img_rgb = load_rgb(query_path)
        queries_by_feature = extract_query_features(
            img_rgb=img_rgb,
            feature_types=feature_types,
        )

        # compute distances only for the existing subset of candidate IDs
        dist_dict, canonical_ids = distances_all_features_subset(
            run_dir=run_dir,
            queries_by_feature=queries_by_feature,
            subset_ids=reference_ids,
            feature_types=feature_types,
            id_to_vec_maps=id_to_vec_maps,
        )

        # ensure alignment consistency across all queries
        if canonical_ids != reference_ids:
            raise ValueError("Canonical candidate ids differ across query runs")

        used_features = set(dist_dict.keys())

        if used_features != reference_features:
            raise ValueError("Used feature sets differ across query runs")

        # compute scores for this query on the shared candidate set
        score_arr = get_score_arr(dist_dict=dist_dict, weights=weights)

        score_arrays.append(score_arr)

    # stack scores per query and aggregate via mean
    stacked_scores = np.stack(score_arrays, axis=0)
    aggregated_scores = np.mean(stacked_scores, axis=0)

    # rank candidates (lower score = better)
    ranked_idxs = np.argsort(aggregated_scores)
    final_k = min(k, aggregated_scores.shape[0])

    top_k = [(reference_ids[idx], aggregated_scores[idx]) for idx in ranked_idxs[:final_k]]

    return top_k, reference_features
