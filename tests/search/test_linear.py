from pathlib import Path

import numpy as np
import pytest

from image_recommender.features.storage import read_validate_shard
from image_recommender.search.linear import LinearSearchBackend


def euclidean_distance(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Returns the euclidean distance between a query and a set of candidates."""
    diff = candidates - query
    return np.sqrt(np.sum(diff**2, axis=1))


@pytest.fixture(scope="module")  # fixture for the LinearSearchBackend
def hsv_pilot_backend() -> LinearSearchBackend:
    """Returns a LinearSearchBackend instance for the pilot dataset."""
    run_dir = Path("data/features/pilot")
    feature_type = "hsv"

    return LinearSearchBackend(
        run_dir=run_dir,
        feature_type=feature_type,
        distance_fn=euclidean_distance,
        k=5,
    )


@pytest.fixture(scope="module")
def sample_query():
    """Returns a feature vector from shard 0 and its corresponding image id."""
    run_dir = Path("data/features/pilot")
    features, ids = read_validate_shard(
        run_dir=run_dir,
        feature_type="hsv",
        shard_id=0,
    )
    return features[0], ids[0]


@pytest.mark.integration  # integration test for LinearSearch
def test_self_match_top1(hsv_pilot_backend, sample_query):
    """Tests that the top-1 result is the same as the query."""
    query, expected_id = sample_query

    ids, dists = hsv_pilot_backend.search(query)

    assert ids[0] == expected_id
    assert dists[0] < 1e-10


@pytest.mark.integration
def test_result_length_and_sorted(hsv_pilot_backend, sample_query):
    """Tests that the top-k results are sorted and have the correct length."""
    query, _ = sample_query

    ids, dists = hsv_pilot_backend.search(query)

    assert len(ids) == hsv_pilot_backend.k
    assert dists.dtype == np.float32
    assert np.all(np.diff(dists) >= 0)


@pytest.mark.integration
def test_dimensionality_mismatch(hsv_pilot_backend):
    """Tests that a dimensionality mismatch raises an error."""
    with pytest.raises(ValueError):
        hsv_pilot_backend.search(np.array([1.0, 2.0]))


@pytest.mark.integration
def test_zero_query_raises(hsv_pilot_backend):
    """Tests that a zero query raises an error."""
    dim = read_validate_shard(
        run_dir=Path("data/features/pilot"),
        feature_type="hsv",
        shard_id=0,
    )[
        0
    ].shape[1]

    with pytest.raises(ValueError):
        hsv_pilot_backend.search(np.zeros(dim, dtype=np.float32))


@pytest.mark.integration
def test_invalid_k_raises():
    """Tests that an invalid k raises an error."""
    with pytest.raises(ValueError):
        LinearSearchBackend(
            run_dir=Path("data/features/pilot"),
            feature_type="hsv",
            distance_fn=euclidean_distance,
            k=0,
        )


@pytest.mark.integration
def test_anomaly_handling(sample_query):
    """Tests that +inf and NaN distances are ignored and do not break search."""
    query, _ = sample_query

    def broken_distance(q: np.ndarray, c: np.ndarray) -> np.ndarray:
        d = euclidean_distance(q, c)
        d[0] = np.nan  # first candidate is invalid (NaN)
        d[1] = np.inf  # second candidate is invalid (+inf)
        return d

    backend = LinearSearchBackend(
        run_dir=Path("data/features/pilot"),
        feature_type="hsv",
        distance_fn=broken_distance,
        k=3,
    )

    ids, dists = backend.search(query)

    assert len(ids) == 3
    assert np.all(np.isfinite(dists))  # Are all distances finite after anomaly handling?


@pytest.mark.integration
def test_mmap_support(sample_query):
    """Tests that mmap support works correctly."""
    query, _ = sample_query

    backend = LinearSearchBackend(
        run_dir=Path("data/features/pilot"),
        feature_type="hsv",
        distance_fn=euclidean_distance,
        k=3,
        mmap=True,
    )

    ids, _ = backend.search(query)

    assert len(ids) == 3
