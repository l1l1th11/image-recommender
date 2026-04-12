from pathlib import Path

import numpy as np
import pytest

from image_recommender.features.storage import read_validate_shard
from image_recommender.io.img_loader import load_rgb
from image_recommender.metrics.chi import chi_distance_to_many
from image_recommender.metrics.cosine import cosine_distance_to_many
from image_recommender.recommender.query_helpers import (
    align_distances,
    distances_all_features,
    distances_all_features_subset,
    distances_per_feature,
    extract_query_features,
    get_score_arr,
    load_canonical_ids,
)


def test_align_distances():
    canonical_ids = [3, 2, 1]

    # ensure correct alignment (order & values) & dtype
    backend_ids_1 = [1, 2, 3]
    backend_distances_1 = np.array([10, 5, 2])

    aligned_distances_1 = align_distances(
        canonical_ids=canonical_ids,
        backend_ids=backend_ids_1,
        backend_distances=backend_distances_1,
    )

    assert np.array_equal(aligned_distances_1, np.array([2, 5, 10], dtype=np.float32))
    assert aligned_distances_1.dtype == np.float32

    # check duplicate backend ids raise
    backend_ids_2 = [1, 1, 3]
    backend_distances_2 = np.array([1, 7, 9])

    with pytest.raises(ValueError):
        align_distances(
            canonical_ids=canonical_ids,
            backend_ids=backend_ids_2,
            backend_distances=backend_distances_2,
        )

    # check missing ids raise
    backend_ids_3 = [1, 3]
    backend_distances_3 = np.array([6, 8])

    with pytest.raises(ValueError):
        align_distances(
            canonical_ids=canonical_ids,
            backend_ids=backend_ids_3,
            backend_distances=backend_distances_3,
        )

    # check extra ids raise
    backend_ids_4 = [1, 2, 3, 4]
    backend_distances_4 = np.array([8, 4, 2, 5])

    with pytest.raises(ValueError):
        align_distances(
            canonical_ids=canonical_ids,
            backend_ids=backend_ids_4,
            backend_distances=backend_distances_4,
        )


@pytest.mark.integration
def test_distances_per_feature_hsv():
    # setup
    run_dir = Path("data/samples")

    # load features and ids
    features_hsv, ids_hsv = read_validate_shard(run_dir=run_dir, feature_type="hsv", shard_id=0)

    # pick query vector
    query_hsv = features_hsv[0]
    query_id_hsv = ids_hsv[0]

    # get aligned distance array
    aligned_distances_hsv = distances_per_feature(
        run_dir=run_dir, feature_type="hsv", distance_fn=chi_distance_to_many, query=query_hsv
    )

    # check length
    assert aligned_distances_hsv.shape[0] == features_hsv.shape[0]

    # check dtype
    assert aligned_distances_hsv.dtype == np.float32

    # check query is best match
    idx_hsv = np.argmin(aligned_distances_hsv)
    canonical_ids_hsv = load_canonical_ids(run_dir=run_dir, feature_type="hsv")

    assert query_id_hsv == canonical_ids_hsv[idx_hsv]

    # check determinism
    aligned_distances_hsv_2 = distances_per_feature(
        run_dir=run_dir, feature_type="hsv", distance_fn=chi_distance_to_many, query=query_hsv
    )
    assert np.array_equal(aligned_distances_hsv, aligned_distances_hsv_2)


@pytest.mark.integration
def test_distances_per_feature_embedding():
    # setup
    run_dir = Path("data/samples")

    # load features and ids
    features_embedding, ids_embedding = read_validate_shard(
        run_dir=run_dir, feature_type="embedding", shard_id=0
    )

    # pick query vector
    query_embedding = features_embedding[0]
    query_id_embedding = ids_embedding[0]

    # get aligned distance array
    aligned_distances_embedding = distances_per_feature(
        run_dir=run_dir,
        feature_type="embedding",
        distance_fn=cosine_distance_to_many,
        query=query_embedding,
    )

    # check length
    assert aligned_distances_embedding.shape[0] == features_embedding.shape[0]

    # check dtype
    assert aligned_distances_embedding.dtype == np.float32

    # check query is best match
    idx_embedding = np.argmin(aligned_distances_embedding)
    canonical_ids_embedding = load_canonical_ids(run_dir=run_dir, feature_type="embedding")

    assert query_id_embedding == canonical_ids_embedding[idx_embedding]

    # check determinism
    aligned_distances_embedding_2 = distances_per_feature(
        run_dir=run_dir,
        feature_type="embedding",
        distance_fn=cosine_distance_to_many,
        query=query_embedding,
    )
    assert np.array_equal(aligned_distances_embedding, aligned_distances_embedding_2)


@pytest.mark.integration
def test_distances_all_features_subset():
    run_dir = Path("data/samples")
    query_path = Path("data/samples/image_0007.jpeg")
    subset_ids = [0, 1, 2]

    # load image
    img_rgb = load_rgb(query_path)

    # extract query vectors
    queries_by_feature = extract_query_features(
        img_rgb=img_rgb,
        feature_types=["hsv", "embedding"],
    )

    # compute distance dict for subset ids
    dist_dict = distances_all_features_subset(
        run_dir=run_dir,
        queries_by_feature=queries_by_feature,
        subset_ids=subset_ids,
        feature_types=["hsv", "embedding"],
    )

    # ensure keys match expected
    assert set(dist_dict.keys()) == {"hsv", "embedding"}

    # ensure output has correct structure
    for distances in dist_dict.values():
        assert isinstance(distances, np.ndarray)
        assert distances.ndim == 1
        assert len(distances) == len(subset_ids)
        assert np.isfinite(distances).all()


@pytest.mark.integration
def test_get_score_arr():
    # setup
    run_dir = Path("data/samples")

    # load features and ids
    features_hsv, _ = read_validate_shard(run_dir=run_dir, feature_type="hsv", shard_id=0)
    features_embedding, _ = read_validate_shard(
        run_dir=run_dir, feature_type="embedding", shard_id=0
    )

    # pick query vector
    query_hsv = features_hsv[0]
    query_embedding = features_embedding[0]

    # build query dict
    queries_by_feature = {"hsv": query_hsv, "embedding": query_embedding}

    # get distance dict
    dist_dict = distances_all_features(run_dir=run_dir, queries_by_feature=queries_by_feature)

    # get score array
    score_arr = get_score_arr(dist_dict=dist_dict)

    # check length
    n_candidates = next(iter(dist_dict.values())).shape[0]
    assert len(score_arr) == n_candidates

    # check dtype
    assert score_arr.dtype == np.float32

    # ensure no NaN leaked from scoring
    assert np.all(np.isfinite(score_arr))

    # get score array for hsv only
    weights = {"hsv": 1.0, "embedding": 0.0}
    score_arr_2 = get_score_arr(dist_dict, weights)

    # check result equals hsv only scoring
    score_arr_2_hsv_only = get_score_arr({"hsv": dist_dict["hsv"]})
    assert np.allclose(score_arr_2, score_arr_2_hsv_only)

    # check determinism
    score_arr_3 = get_score_arr(dist_dict=dist_dict)
    assert np.array_equal(score_arr, score_arr_3)

    # check empty dict raises
    with pytest.raises(ValueError):
        get_score_arr(dist_dict={})
