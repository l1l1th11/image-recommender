import time
from pathlib import Path

import numpy as np

from image_recommender.config import DEFAULT_K_CANDIDATES
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
    subset_ids: list[int] | None = None,
    annoy_backend: AnnoySearchBackend | None = None,
    id_to_vec_maps=None,
) -> tuple[np.ndarray, list[int], set[str], dict[str, float]]:
    """
    Compute per candidate scores for a single query including detailed timing.

    Executes full query pipeline:
    1. Load image
    2. Extract query features
    3. Generate candidate subset (annoy) or use full dataset (linear)
    4. Compute distances
    5. Aggregate scores

    Input:
    query_path: Path to the query image
    run_dir: Root directory containing feature shards
    feature_types: Optional subset of features to use
    weights: Optional feature weights for score aggregation
    backend: "linear" (full scan) or "annoy" (subset based search)
    k_candidates: Number of candidates retrieved by annoy (if applicable)
    subset_ids: Optional externally provided candidate subset
    annoy_backend: Optional pre initialized annoy backend
    id_to_vec_maps: Precomputed mappings {feature_type: {image_id: feature_vector}} required for annoy backend

    Output:
        score_arr: Array of shape (N,) with scores aligned to canonical_ids
        canonical_ids: Candidate IDs aligned to score_arr
        used_features: Set of features actually used
        timings: Dictionary containing per stage execution times

    Raises:
        ValueError: If required features are missing or no valid candidates remain

    Notes:
        - Annoy mode uses subset based distance computation via id to vector mapping
        - Linear mode computes distances against full dataset
    """
    # timing container
    timings = {}
    t_total_start = time.perf_counter()

    # load image
    t0 = time.perf_counter()
    img_rgb = load_rgb(path=query_path)
    timings["load_image"] = time.perf_counter() - t0

    # extract query features
    t0 = time.perf_counter()
    queries_by_feature = extract_query_features(img_rgb=img_rgb, feature_types=feature_types)
    timings["feature_extraction"] = time.perf_counter() - t0

    # annoy based search
    if backend == "annoy":

        # ensure embedding is available
        if "embedding" not in queries_by_feature:
            raise ValueError("Annoy backend requires embedding feature")

        # apply default candidate size
        if k_candidates is None:
            k_candidates = DEFAULT_K_CANDIDATES

        # get embedding query
        query_embedding = queries_by_feature["embedding"]

        # initialize annoy backend if not provided
        if annoy_backend is None:
            annoy_backend = AnnoySearchBackend(
                run_dir=run_dir,
                feature_type="embedding",
                k=k_candidates,
            )

        # step 1: candidate generation
        t0 = time.perf_counter()

        if subset_ids is None:
            subset_ids_arr, _ = annoy_backend.search(query_embedding)
            candidate_ids = subset_ids_arr.tolist()
        else:
            candidate_ids = subset_ids

        timings["candidate_generation"] = time.perf_counter() - t0

        # step 2: distance computation on subset
        if id_to_vec_maps is None:
            raise ValueError("id_to_vec_maps must be provided for annoy backend")

        t0 = time.perf_counter()

        # explicitly pass mapping to subset function
        dist_dict, canonical_ids = distances_all_features_subset(
            run_dir=run_dir,
            queries_by_feature=queries_by_feature,
            subset_ids=candidate_ids,
            feature_types=feature_types,
            id_to_vec_maps=id_to_vec_maps,
        )

        timings["distance_computation"] = time.perf_counter() - t0

        # collect features actually used
        used_features = set(dist_dict.keys())

        if not used_features:
            raise ValueError("No features available after distance computation")

        # step 3: scoring
        t0 = time.perf_counter()
        score_arr = get_score_arr(dist_dict=dist_dict, weights=weights)
        timings["scoring"] = time.perf_counter() - t0

        # total time
        timings["total"] = time.perf_counter() - t_total_start

        return score_arr, canonical_ids, used_features, timings

    # linear search

    # compute distances
    t0 = time.perf_counter()
    dist_dict = distances_all_features(
        run_dir=run_dir,
        queries_by_feature=queries_by_feature,
        feature_types=feature_types,
    )
    timings["distance_computation"] = time.perf_counter() - t0

    used_features = set(dist_dict.keys())

    if not used_features:
        raise ValueError("No features available after distance computation")

    # select reference feature for canonical ids
    reference_feature = sorted(used_features)[0]
    canonical_ids = load_canonical_ids(run_dir=run_dir, feature_type=reference_feature)

    # compute scores
    t0 = time.perf_counter()
    score_arr = get_score_arr(dist_dict=dist_dict, weights=weights)
    timings["scoring"] = time.perf_counter() - t0

    # total time linear
    timings["total"] = time.perf_counter() - t_total_start

    return score_arr, canonical_ids, used_features, timings


def _compute_full_scores_from_features(
    queries_by_feature: dict[str, np.ndarray],
    run_dir: Path | str,
    feature_types: list[str] | None = None,
    weights: dict[str, float] | None = None,
    backend: str = "linear",
    k_candidates: int | None = None,
    subset_ids: list[int] | None = None,
    annoy_backend: AnnoySearchBackend | None = None,
    id_to_vec_maps=None,
) -> tuple[np.ndarray, list[int], set[str]]:
    """
    Variant of _compute_full_scores that bypasses image loading & feature extraction.
    Used for controlled evaluation with precomputed query features.
    """

    # annoy
    if backend == "annoy":
        if "embedding" not in queries_by_feature:
            raise ValueError("Annoy backend requires embedding feature")

        if k_candidates is None:
            k_candidates = DEFAULT_K_CANDIDATES

        query_embedding = queries_by_feature["embedding"]

        if annoy_backend is None:
            annoy_backend = AnnoySearchBackend(
                run_dir=run_dir,
                feature_type="embedding",
                k=k_candidates,
            )

        if subset_ids is None:
            subset_ids_arr, _ = annoy_backend.search(query_embedding)
            candidate_ids = subset_ids_arr.tolist()
        else:
            candidate_ids = subset_ids

        # enforce optimized path
        if id_to_vec_maps is None:
            raise ValueError("id_to_vec_maps must be provided for annoy backend")

        dist_dict, canonical_ids = distances_all_features_subset(
            run_dir=run_dir,
            queries_by_feature=queries_by_feature,
            subset_ids=candidate_ids,
            feature_types=feature_types,
            id_to_vec_maps=id_to_vec_maps,
        )

        used_features = set(dist_dict.keys())

        if not used_features:
            raise ValueError("No features available after distance computation")

        score_arr = get_score_arr(dist_dict=dist_dict, weights=weights)

        return score_arr, canonical_ids, used_features

    # linear

    dist_dict = distances_all_features(
        run_dir=run_dir,
        queries_by_feature=queries_by_feature,
        feature_types=feature_types,
    )

    used_features = set(dist_dict.keys())

    if not used_features:
        raise ValueError("No features available after distance computation")

    reference_feature = sorted(used_features)[0]
    canonical_ids = load_canonical_ids(run_dir=run_dir, feature_type=reference_feature)

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
    annoy_backend: AnnoySearchBackend | None = None,
    id_to_vec_maps: dict[str, dict[int, np.ndarray]] | None = None,
) -> tuple[list[tuple[int, float]], set[str]]:
    """
    Runs a single image query and returns the top k most similar results, as well as used feature types.
    Requires precomputed id to vector mappings for annoy backend for efficient subset distance computation.

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
    score_arr, canonical_ids, used_features, _ = _compute_full_scores(
        query_path=query_path,
        run_dir=run_dir,
        feature_types=feature_types,
        weights=weights,
        backend=backend,
        k_candidates=k_candidates,
        annoy_backend=annoy_backend,
        id_to_vec_maps=id_to_vec_maps,
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
