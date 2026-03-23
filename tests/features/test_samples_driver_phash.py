import numpy as np
import pytest

from image_recommender.features.samples_driver_phash import compute_topk


def phash_helper():
    vector_1 = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
    vector_2 = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.uint8)
    vector_3 = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)
    vector_4 = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8)

    vectors = np.vstack([vector_1, vector_2, vector_3, vector_4])
    ids = ["1", "2", "3", "4"]

    return vectors, ids


def test_length_neighbors():
    vectors, ids = phash_helper()
    results = compute_topk(ids=ids, phashes=vectors, k=2)

    # ensure length of candidates and neighbors is consistent
    assert len(results) == 4
    for n_neighbors in results:
        assert len(n_neighbors) == 2


def test_self_top_neighbor():
    vectors, ids = phash_helper()
    results = compute_topk(ids=ids, phashes=vectors, k=2)

    # ensure self is nearest neighbor
    for query_id, n_neighbors in zip(ids, results, strict=True):
        top_id, top_dist = n_neighbors[0]
        assert top_dist == 0
        assert top_id == query_id


def test_distances_sorted_ascending():
    vectors, ids = phash_helper()
    results = compute_topk(ids=ids, phashes=vectors, k=2)

    # ensure distances are sorted ascending
    for n_neighbors in results:
        distances = [dist for _, dist in n_neighbors]
        # compare adjacent elements
        for a, b in zip(distances, distances[1:], strict=False):
            assert a <= b


def test_invalid_k():
    vectors, ids = phash_helper()

    # check k <= 0 raises ValueError
    with pytest.raises(ValueError):
        compute_topk(ids=ids, phashes=vectors, k=0)
    with pytest.raises(ValueError):
        compute_topk(ids=ids, phashes=vectors, k=-1)


def test_clamp_k_to_n():
    vectors, ids = phash_helper()
    results = compute_topk(ids=ids, phashes=vectors, k=8)

    # check k > n is clamped to n
    n = len(ids)

    for n_neighbors in results:
        assert len(n_neighbors) == n
