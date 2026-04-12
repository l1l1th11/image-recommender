from pathlib import Path

import numpy as np

from image_recommender.io.img_loader import load_rgb
from image_recommender.recommender.query_helpers import (
    distances_all_features,
    distances_all_features_subset,
    extract_query_features,
    get_score_arr,
    load_canonical_ids,
)
from image_recommender.search.annoy import AnnoySearchBackend
from image_recommender.util.logs import get_logger

logger = get_logger(__name__)


def _compute_full_scores(
    query_path: str | Path,
    run_dir: Path | str,
    feature_types: list[str] | None = None,
    weights: dict[str, float] | None = None,
    backend: str = "linear",
    k_candidates: int | None = None,
) -> tuple[np.ndarray, list[int], set[str]]:
    """
    Internal helper to compute full aligned score array for a single query.

    Output:
        score_arr: (N,) aligned scores
        canonical_ids: list of candidate ids aligned to canonical candidate id order
        used_features: set of features actually used
    """
    # load image
    img_rgb = load_rgb(path=query_path)

    # extract query features
    queries_by_feature = extract_query_features(img_rgb=img_rgb, feature_types=feature_types)

    # annoy search
    if backend == "annoy":
        if "embedding" not in queries_by_feature:
            raise ValueError("Annoy backend requires embedding feature")

        if k_candidates is None:
            raise ValueError("k_candidates must be provided for annoy backend")

        # get embedding query
        query_embedding = queries_by_feature["embedding"]

        # run annoy to get candidate subset
        annoy_backend = AnnoySearchBackend(
            run_dir=run_dir,
            feature_type="embedding",
            k=k_candidates,
        )

        subset_ids, _ = annoy_backend.search(query_embedding)

        # define subset ids order as canonical order for this mode
        canonical_ids = subset_ids.tolist()

        # compute distances only for subset
        dist_dict = distances_all_features_subset(
            run_dir=run_dir,
            queries_by_feature=queries_by_feature,
            subset_ids=canonical_ids,
            feature_types=feature_types,
        )

        used_features = set(dist_dict.keys())

        if not used_features:
            raise ValueError("No features available after distance computation")

        # compute scores
        score_arr = get_score_arr(dist_dict=dist_dict, weights=weights)

        return score_arr, canonical_ids, used_features

    # linear search

    # compute distances
    dist_dict = distances_all_features(
        run_dir=run_dir,
        queries_by_feature=queries_by_feature,
        feature_types=feature_types,
    )

    used_features = set(dist_dict.keys())

    if not used_features:
        raise ValueError("No features available after distance computation")

    # select reference feature for canonical ids
    reference_feature = sorted(used_features)[0]
    canonical_ids = load_canonical_ids(run_dir=run_dir, feature_type=reference_feature)

    # compute scores
    score_arr = get_score_arr(dist_dict=dist_dict, weights=weights)

    return score_arr, canonical_ids, used_features


def single_image_query(
    query_path: str | Path,
    run_dir: Path | str,
    k: int,
    feature_types: list[str] | None = None,
    weights: dict[str, float] | None = None,
    backend: str = "linear",
    k_candidates: int | None = None,
) -> tuple[list[tuple[int, float]], set[str]]:
    """
    Runs a single image query and returns the top k most similar results, as well as used feature types.

    Input:
        query_path: Path to query image
        run_dir: Directory containing feature folders
        k: Number of top results to return
        feature_types: Optional subset of feature types to process
        weights: Optional weights (must match keys, sum to appr. 1)
        backend: Linear search per deafult, or annoy based
        k_candidates: Size of candidate subset retrieved by annoy (required for annoy backend)

    Output:
        top_k: List of (image_id, score) pairs sorted ascending (best match first)
        used_features: Set of actually used feature types

    Raises:
        ValueError if no features or candidates are available

    Notes:
        Loads image from path
        Extracts query features
        Computes per candidate scores across available features
        Ranks candidates by score (ascending)
        Returns top-k (id, score) pairs
    """
    # get aligned score array, canonical id list and used features set
    score_arr, canonical_ids, used_features = _compute_full_scores(
        query_path=query_path,
        run_dir=run_dir,
        feature_types=feature_types,
        weights=weights,
        backend=backend,
        k_candidates=k_candidates,
    )

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

    return top_k, used_features
