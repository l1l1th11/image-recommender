from pathlib import Path

import numpy as np
import pytest

from image_recommender.recommender.multi_image_query import multi_image_query
from image_recommender.recommender.query_helpers import load_canonical_ids
from image_recommender.recommender.single_image_query import _compute_full_scores


@pytest.mark.integration
def test_basic_function():
    # setup
    run_dir = Path("data/samples")
    k = 5
    query_paths = [Path("data/samples/image_0007.jpeg"), Path("data/samples/image_1306.jpg")]

    # compute top k
    top_k, used_features = multi_image_query(query_paths=query_paths, run_dir=run_dir, k=k)

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

    # check ranking correctness
    scores = [score for _, score in top_k]
    assert scores == sorted(scores)

    assert isinstance(used_features, set)


@pytest.mark.integration
def test_aggregation():
    # setup
    run_dir = Path("data/samples")
    k = 5
    query_path_1 = Path("data/samples/image_0007.jpeg")
    query_path_2 = Path("data/samples/image_1306.jpg")
    query_paths = [query_path_1, query_path_2]

    # run multi image query
    top_k, _ = multi_image_query(query_paths=query_paths, run_dir=run_dir, k=k)

    # compute scores manually
    score_arr_1, canonical_ids, _ = _compute_full_scores(query_path=query_path_1, run_dir=run_dir)
    score_arr_2, _, _ = _compute_full_scores(query_path=query_path_2, run_dir=run_dir)

    stacked_scores = np.stack([score_arr_1, score_arr_2], axis=0)
    aggregated_scores = np.mean(stacked_scores, axis=0)

    # build lookup (id -> aggregated score)
    score_by_id = dict(zip(canonical_ids, aggregated_scores, strict=True))

    # compare top k results
    for candidate_id, score in top_k:
        assert np.isclose(score, score_by_id[candidate_id])


@pytest.mark.integration
def test_multi_query_determinism():
    run_dir = Path("data/samples")
    k = 5
    query_paths = [Path("data/samples/image_0007.jpeg"), Path("data/samples/image_1306.jpg")]

    result1 = multi_image_query(query_paths, run_dir, k)
    result2 = multi_image_query(query_paths, run_dir, k)

    assert result1 == result2
