from pathlib import Path

import numpy as np
import pytest

from image_recommender.recommender.query_helpers import (
    SUPPORTED_FEATURES,
    load_canonical_ids,
)
from image_recommender.recommender.single_image_query import single_image_query


@pytest.mark.integration
def test_single_image_query():
    # setup
    run_dir = Path("data/samples")
    k = 3
    query_path = Path("data/samples/image_0007.jpeg")

    # discover available features in run_dir & select one
    feature_dirs = sorted(
        [
            feature_dir
            for feature_dir in run_dir.iterdir()
            if feature_dir.is_dir() and feature_dir.name in SUPPORTED_FEATURES
        ]
    )
    feature_type = feature_dirs[0].name

    # compute canonical ids and candidates
    canonical_ids = load_canonical_ids(run_dir=run_dir, feature_type=feature_type)
    n_candidates = len(canonical_ids)

    # compute top k
    top_k = single_image_query(query_path=query_path, run_dir=run_dir, k=k)

    # check length is correct
    assert len(top_k) == min(k, n_candidates)

    # check structure
    for pair in top_k:
        assert isinstance(pair[0], int)
        assert isinstance(pair[1], (np.floating, float))

    # check ids are valid
    for pair in top_k:
        assert pair[0] in canonical_ids

    # check sorting
    scores = []

    for pair in top_k:
        scores.append(pair[1])

    assert scores == sorted(scores)

    # check self is best match
    top_id, _ = top_k[0]

    assert top_id == 0  # corresponds to image_007.jpeg

    # check determinism
    top_k_2 = single_image_query(query_path=query_path, run_dir=run_dir, k=k)

    assert top_k == top_k_2
