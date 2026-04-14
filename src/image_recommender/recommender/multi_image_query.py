from pathlib import Path

import numpy as np

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
    Runs a multi image query by aggregating aligned per query score arrays via mean.

    Input:
        query_paths: List of query image paths
        run_dir: Directory containing feature folders
        k: Number of top results to return
        feature_types: Optional subset of feature types to process
        weights: Optional weights (must match keys, sum to appr. 1)
        k_candidates: Size of candidate subset retrieved by annoy (required for annoy backend)
        id_to_vec_maps: Precomputed mappings {feature_type: {image_id: feature_vector}} required for annoy backend

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
    first_query = query_paths[0]

    # generate subset from first query
    score_arr, canonical_ids, used_features, _ = _compute_full_scores(
        query_path=first_query,
        run_dir=run_dir,
        feature_types=feature_types,
        weights=weights,
        backend=backend,
        k_candidates=k_candidates,
        annoy_backend=annoy_backend,
        id_to_vec_maps=id_to_vec_maps,
    )

    score_arrays = [score_arr]
    reference_ids = canonical_ids
    reference_features = used_features

    # process remaining queries
    for query_path in query_paths[1:]:

        # compute scores based on first queries subset
        score_arr, canonical_ids, used_features, _ = _compute_full_scores(
            query_path=query_path,
            run_dir=run_dir,
            feature_types=feature_types,
            weights=weights,
            backend=backend,
            k_candidates=k_candidates,
            annoy_backend=annoy_backend,
            subset_ids=reference_ids if backend == "annoy" else None,
            id_to_vec_maps=id_to_vec_maps,
        )

        # canonical id check
        if canonical_ids != reference_ids:
            raise ValueError("Canonical candidate ids differ across query runs")

        # feature consistency check
        if used_features != reference_features:
            raise ValueError("Used feature sets differ across query runs")

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
