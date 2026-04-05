import numpy as np
import pytest

from image_recommender.recommender.single_image_query import align_distances


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
