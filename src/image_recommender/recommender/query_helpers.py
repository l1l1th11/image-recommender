from collections.abc import Callable
from pathlib import Path

import numpy as np

from image_recommender.constants import SUPPORTED_FEATURES
from image_recommender.features.embedding import extract_embedding
from image_recommender.features.hsv import hsv_features
from image_recommender.features.phash import extract_phash
from image_recommender.features.storage import read_validate_shard
from image_recommender.metrics.chi import chi_distance_to_many
from image_recommender.metrics.cosine import cosine_distance_to_many
from image_recommender.metrics.hamming import hamming_distance_to_many
from image_recommender.recommender.scoring import compute_scores
from image_recommender.search.linear import LinearSearchBackend
from image_recommender.util.logs import get_logger

logger = get_logger(__name__)


def load_canonical_ids(run_dir: Path | str, feature_type: str) -> list[int]:
    # get sorted shard dirs
    data = Path(run_dir) / Path(feature_type)
    shard_dirs = [shard_dir for shard_dir in sorted(data.glob("shard_*")) if shard_dir.is_dir()]

    canonical_ids = []

    # read ids per shard
    for shard_dir in shard_dirs:
        # extract index
        shard_id = int(shard_dir.name.split("_")[1])
        _, ids_list = read_validate_shard(
            run_dir=run_dir, feature_type=feature_type, shard_id=shard_id
        )
        canonical_ids.extend(ids_list)

    # guard against duplicates
    if len(canonical_ids) != len(set(canonical_ids)):
        raise ValueError("Canonical id set contains duplicates")

    return canonical_ids


def align_distances(
    canonical_ids: list[int], backend_ids: list[int], backend_distances: np.ndarray
) -> np.ndarray:
    # ensure no duplicate ids in backend
    if len(backend_ids) != len(set(backend_ids)):
        raise ValueError("Backend id set contains duplicates")

    # ensure backend and canonical ids have equal coverage
    if set(backend_ids) != set(canonical_ids):
        raise ValueError("Backend and canonical id sets are not equal")

    # align ids & distances within dict
    distance_by_id = dict(zip(backend_ids, backend_distances, strict=True))

    # build aligned distance array
    aligned_distances = [distance_by_id[id] for id in canonical_ids]

    return np.array(aligned_distances, dtype=np.float32)


def distances_per_feature(
    run_dir: Path | str,
    feature_type: str,
    distance_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    query: np.ndarray,
) -> np.ndarray:
    """
    Compute distances from a single query to all candidates for one feature type.

    Input:
        run_dir: Directory containing feature folders
        feature_type: Feature type to process ("hsv", "embedding", "phash")
        distance_fn: Function calculating distances between query (D,) and candidates (N, D)
        query: Precomputed query vector of shape (D,)

    Output:
        distances (N,): Distances aligned to canonical candidate id order for selected feature

    Raises:
        ValueError: If feature data is missing, inconsistent, or distance computation fails

    Notes:
        - Distances are computed via search backend, then realigned to canonical id order
    """
    # get canonical ids
    canonical_ids = load_canonical_ids(run_dir=run_dir, feature_type=feature_type)

    # instantiate search backend
    backend = LinearSearchBackend(
        run_dir=run_dir, feature_type=feature_type, distance_fn=distance_fn, k=1
    )  # k is not used by search_all

    # get all ids & distances for one feature
    backend_ids, backend_distances = backend.search_all(query=query)

    # align distances
    aligned_distances = align_distances(
        canonical_ids=canonical_ids, backend_ids=backend_ids, backend_distances=backend_distances
    )

    return aligned_distances


def distances_all_features(
    run_dir: Path | str,
    queries_by_feature: dict[str, np.ndarray],
    feature_types: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """
    Computes distances from a single query to all candidates for all available feature types.

    Input:
        run_dir: Directory containing feature folders
        queries_by_feature: {feature_type: query_vector (D,)} with precomputed features
        feature_types: Optional subset of feature types to process

    Output:
        {feature_type: distances (N,)} where each array is aligned to canonical candidate id order

    Raises:
        ValueError: If no valid feature types are available after filtering

    Notes:
        - Only features present in both run_dir and queries_by_feature are used
        - Missing or failing features are skipped with a warning
    """
    # discover available features from run_dir
    data = Path(run_dir)
    features_run_dir = set(
        feature_dir.name for feature_dir in data.iterdir() if feature_dir.is_dir()
    )

    # get available features from queries
    features_query = set(queries_by_feature.keys())

    # proceed with available features only
    available_features = features_query & features_run_dir

    # check for mismatch
    if features_query != available_features:

        # warn if requested features are not available
        logger.warning(
            f"One or more requested features aren't available, proceeding with {available_features}"
        )

    # filter by selected feature types
    if feature_types is None:
        features_to_process = available_features

    else:
        features_to_process = set(feature_types) & available_features

    # check if there are features to process
    if not features_to_process:
        raise ValueError("No available features to process")

    # build distance function dict
    distance_fn_dict = {
        "hsv": chi_distance_to_many,
        "embedding": cosine_distance_to_many,
        "phash": hamming_distance_to_many,
    }

    dist_dict = {}

    for feature in sorted(features_to_process):
        # get required query and distance function
        query = queries_by_feature[feature]
        distance_fn = distance_fn_dict[feature]

        # calculate distance array
        try:
            aligned_distances = distances_per_feature(
                run_dir=run_dir, feature_type=feature, distance_fn=distance_fn, query=query
            )

            # append distances for each feature
            dist_dict[feature] = aligned_distances

        # skip and log on error
        except ValueError as e:
            logger.warning(f"Distance computation failed for {feature}: {e}", exc_info=True)

            continue

    return dist_dict


def distances_all_features_subset(
    run_dir: Path | str,
    queries_by_feature: dict[str, np.ndarray],
    subset_ids: list[int],
    feature_types: list[str] | None = None,
    id_to_vec_maps: dict[str, dict[int, np.ndarray]] | None = None,
) -> tuple[dict[str, np.ndarray], list[int]]:
    """
    Compute distances from one query to a candidate subset for all usable feature types.

    Input:
        run_dir: Root directory containing feature folders
        queries_by_feature: Mapping {feature_type: query_vector}
        subset_ids: Candidate ids defining both subset and target order
        feature_types: Optional subset of feature types to process
        id_to_vec_maps: Prebuilt mappings {feature_type: {image_id: feature_vector}}

    Output:
        dist_dict: Mapping {feature_type: distances} where each distance array is aligned to returned shared_subset_ids
        shared_subset_ids: Candidate ids that are available across all processed feature types preserving original order from subset_ids

    Raises:
        ValueError: If no usable features are available, mappings are missing, or no common candidate ids remain after feature filtering

    Notes:
        - This is the optimized subset path used for ANN re ranking
        - Requires prebuilt id to vector mappings
        - Preserves original order of subset_ids
    """
    run_dir = Path(run_dir)

    # ensure optimized path is used
    if id_to_vec_maps is None:
        raise ValueError("id_to_vec_maps must be provided for subset distance computation")

    # discover available features from run_dir
    features_run_dir = {
        feature_dir.name for feature_dir in run_dir.iterdir() if feature_dir.is_dir()
    }

    # features available from query extraction
    features_query = set(queries_by_feature.keys())

    # proceed with intersection only
    available_features = features_query & features_run_dir

    # filter by selected feature types (if provided)
    if feature_types is None:
        features_to_process = available_features
    else:
        features_to_process = set(feature_types) & available_features

    # guard against empty feature set
    if not features_to_process:
        raise ValueError("No available features to process")

    # map feature type to corresponding distance function
    distance_fn_dict = {
        "hsv": chi_distance_to_many,
        "embedding": cosine_distance_to_many,
        "phash": hamming_distance_to_many,
    }

    # containers for results and intermediate filtering
    dist_dict: dict[str, np.ndarray] = {}
    valid_ids_per_feature: dict[str, set[int]] = {}
    id_to_vec_per_feature: dict[str, dict[int, np.ndarray]] = {}

    # step 1: filter subset per feature
    for feature in sorted(features_to_process):

        # ensure mapping exists for feature
        if feature not in id_to_vec_maps:
            raise ValueError(f"Missing ID-to-vector mapping for feature '{feature}'")

        id_to_vec = id_to_vec_maps[feature]

        # retain only ids present in current feature mapping
        valid_ids = [image_id for image_id in subset_ids if image_id in id_to_vec]

        if not valid_ids:
            raise ValueError(f"No valid ids left for feature '{feature}'")

        # store valid ids and mapping for later use
        valid_ids_per_feature[feature] = set(valid_ids)
        id_to_vec_per_feature[feature] = id_to_vec

    # step 2: compute shared subset across features

    # keep only ids valid for all features
    shared_ids = set.intersection(*valid_ids_per_feature.values())

    if not shared_ids:
        raise ValueError("No common ids across features")

    # preserve original order from subset_ids
    shared_subset_ids = [image_id for image_id in subset_ids if image_id in shared_ids]

    # step 3: compute distances on shared subset
    for feature in sorted(features_to_process):

        id_to_vec = id_to_vec_per_feature[feature]

        # build candidate matrix aligned to shared_subset_ids order
        subset_matrix = np.vstack([id_to_vec[image_id] for image_id in shared_subset_ids])

        # compute distances for current feature
        query = queries_by_feature[feature]
        distance_fn = distance_fn_dict[feature]
        distances = distance_fn(query, subset_matrix)

        # store distances aligned to shared subset
        dist_dict[feature] = distances.astype(np.float32)

    return dist_dict, shared_subset_ids


def get_score_arr(
    dist_dict: dict[str, np.ndarray],
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    """
    Computes a per candidate score array from aligned per feature distance arrays.

    Input:
        dist_dict: per feature distance arrays (same length & order)
        weights: Optional weights (must match keys, sum to appr. 1)

    Output:
        score_arr: Aligned to canonical id order

    Raises:
        ValueError: If no valid features available
    """
    # guard against meaningless scoring results
    if not dist_dict:
        raise ValueError("No valid features available to compute scores")

    # compute scores
    score_arr = compute_scores(dist_dict=dist_dict, weights=weights)

    return score_arr


def extract_query_features(
    img_rgb: np.ndarray, feature_types: list[str] | None = None
) -> dict[str, np.ndarray]:
    """
    Extracts query feature vectors from an RGB image for the specified feature types.

    Input:
        img_rgb: rgb image as a numpy array (prev. validated by image loader)
        feature_types: Optional list of feature types to extract. If None, all supported features are attempted

    Output:
        Dictionary mapping feature type to extracted query vector (1D numpy array)

    Raises:
        ValueError if no valid features can be processed

    Notes:
        Only supported feature types are processed
        Unsupported requested features are ignored with a warning
        If extraction of a feature fails, it is skipped with a warning
    """
    # select features to process
    if feature_types is None:
        features_to_process = SUPPORTED_FEATURES

    else:
        # filter by selected feature types
        features_to_process = set(SUPPORTED_FEATURES) & set(feature_types)

        # warn if requested features are not supported
        if not set(feature_types).issubset(SUPPORTED_FEATURES):
            logger.warning(
                f"One or more requested features aren't supported (supported: {SUPPORTED_FEATURES}), proceeding with {features_to_process}"
            )

    # check if there are features to process
    if not features_to_process:
        raise ValueError("No available features to process")

    # construct queries dict
    queries_by_feature = {}

    # call right extractor
    for feature in sorted(features_to_process):

        try:
            if feature == "hsv":
                query_hsv = hsv_features(img_rgb=img_rgb)
                queries_by_feature["hsv"] = query_hsv

            elif feature == "embedding":
                query_embedding = extract_embedding(img_rgb=img_rgb)
                queries_by_feature["embedding"] = query_embedding

            elif feature == "phash":
                query_phash = extract_phash(img_rgb=img_rgb)
                queries_by_feature["phash"] = query_phash

        # skip features if extraction fails
        except ValueError as e:
            logger.warning(f"Query feature for {feature} could not be extracted, skipping: {e}")

    # check if any queries were extracted
    if not queries_by_feature:
        raise ValueError("No queries could be extracted")

    return queries_by_feature
