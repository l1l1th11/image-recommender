import numpy as np
import pytest

from image_recommender.recommender.scoring import normalize, validate_input


def test_validate_input():
    distances_1 = {"hsv": np.array([1, 2, 3]), "embedding": np.array([4, 5, 6])}
    distances_2 = {}
    distances_3 = {"hsv": np.array([1, 2, 3]), "embedding": np.array([4, 5])}

    # check returned n_candidates has correct value
    n_candidates = validate_input(distances=distances_1)
    assert n_candidates == 3

    # ensure empty dict raises
    with pytest.raises(ValueError):
        validate_input(distances=distances_2)

    # ensure mismatched length raises
    with pytest.raises(ValueError):
        validate_input(distances=distances_3)


def test_normalize():
    dist_array_1 = np.array([1, 2, 3])
    dist_array_2 = np.array([1, 1, 1])
    norm_dist_array_1 = normalize(dist_array_1)
    norm_dist_array_2 = normalize(dist_array_2)

    # check values are in [0, 1] and float
    assert np.all(norm_dist_array_1 >= 0)
    assert np.all(norm_dist_array_1 <= 1)
    assert norm_dist_array_1.dtype == np.float32

    # check order is preserved
    assert np.allclose(norm_dist_array_1, np.array([0.0, 0.5, 1.0], dtype=np.float32))

    # ensure 0 variance is handled
    assert np.allclose(norm_dist_array_2, np.zeros(shape=dist_array_2.shape, dtype=np.float32))
