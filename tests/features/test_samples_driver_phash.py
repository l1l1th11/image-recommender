import numpy as np

from image_recommender.features.samples_driver_phash import compute_topk


def test_length_neighbors():
    vector_1 = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
    vector_2 = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.uint8)
    vector_3 = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)
    vector_4 = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8)

    vectors = np.asarray([vector_1, vector_2, vector_3, vector_4])
    ids = ["1", "2", "3", "4"]

    results = compute_topk(ids=ids, phashes=vectors, k=2)

    # ensure length of candidates and neighbors is consistent
    assert len(results) == 4
    for n_neighbors in results:
        assert len(n_neighbors) == 2
