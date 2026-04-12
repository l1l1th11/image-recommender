from pathlib import Path

import numpy as np
import pytest

from image_recommender.recommender.query_helpers import (
    SUPPORTED_FEATURES,
    load_canonical_ids,
)
from image_recommender.recommender.single_image_query import (
    _compute_full_scores,
    single_image_query,
)


@pytest.mark.integration
def test_single_image_query():
    # setup
    run_dir = Path("data/samples")
    k = 25
    query_path = Path("data/samples/image_0007.jpeg")
    feature_types = ["hsv", "embedding", "nonexistent"]

    # compute top k
    top_k, used_features = single_image_query(
        query_path=query_path, run_dir=run_dir, k=k, feature_types=feature_types
    )

    # ensure features remain for score computation
    assert used_features

    # compute canonical ids and candidates
    reference_feature = sorted(used_features)[0]
    canonical_ids = load_canonical_ids(run_dir=run_dir, feature_type=reference_feature)
    n_candidates = len(canonical_ids)

    # check length is correct
    assert len(top_k) == min(k, n_candidates)

    # check structure
    for pair in top_k:
        assert isinstance(pair[0], int)
        assert isinstance(pair[1], (np.floating, float))

    assert isinstance(used_features, set)

    # check ids are valid
    for pair in top_k:
        assert pair[0] in canonical_ids

    # check returned features are valid
    assert all(f in SUPPORTED_FEATURES for f in used_features)

    # check sorting
    scores = []

    for pair in top_k:
        scores.append(pair[1])

    assert all(scores[i] <= scores[i + 1] for i in range(len(scores) - 1))

    # check self is best match
    top_id, _ = top_k[0]

    assert top_id == 0  # corresponds to image_007.jpeg
    assert top_id in canonical_ids
    assert scores[0] == min(scores)

    # check determinism
    top_k_2, used_features_2 = single_image_query(
        query_path=query_path, run_dir=run_dir, k=k, feature_types=feature_types
    )

    assert top_k == top_k_2
    assert used_features == used_features_2


@pytest.mark.integration
def test_single_image_query_annoy_mode():
    run_dir = Path("data/samples")
    query_path = Path("data/samples/image_0007.jpeg")

    score_arr, ids, used_features = _compute_full_scores(
        query_path=query_path,
        run_dir=run_dir,
        backend="annoy",
        k_candidates=5,
    )

    # check shape
    assert len(score_arr) == len(ids)
    assert len(ids) == 5

    # check types
    assert isinstance(score_arr, np.ndarray)
    assert isinstance(ids, list)
    assert isinstance(used_features, set)

    # check numeric values
    assert np.isfinite(score_arr).all()

    # ensure no duplicates
    assert len(set(ids)) == len(ids)
