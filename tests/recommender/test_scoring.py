import numpy as np
import pytest

from image_recommender.recommender.scoring import (
    filter_0_variance,
    normalize_array,
    normalize_dict,
    validate_input,
)


def test_validate_input():
    dist_dict_1 = {"hsv": np.array([1, 2, 3]), "embedding": np.array([4, 5, 6])}
    dist_dict_2 = {}
    dist_dict_3 = {"hsv": np.array([1, 2, 3]), "embedding": np.array([4, 5])}

    # check returned n_candidates has correct value
    n_candidates = validate_input(dist_dict=dist_dict_1)
    assert n_candidates == 3

    # ensure empty dict raises
    with pytest.raises(ValueError):
        validate_input(dist_dict=dist_dict_2)

    # ensure mismatched length raises
    with pytest.raises(ValueError):
        validate_input(dist_dict=dist_dict_3)


def test_normalize_():
    dist_array_1 = np.array([1, 2, 3])
    dist_array_2 = np.array([1, 1, 1])
    norm_dist_array_1 = normalize_array(dist_array_1)
    norm_dist_array_2 = normalize_array(dist_array_2)

    # check values are in [0, 1] and float
    assert np.all(norm_dist_array_1 >= 0)
    assert np.all(norm_dist_array_1 <= 1)
    assert norm_dist_array_1.dtype == np.float32

    # check order is preserved
    assert np.allclose(norm_dist_array_1, np.array([0.0, 0.5, 1.0], dtype=np.float32))

    # ensure 0 variance is handled
    assert np.allclose(norm_dist_array_2, np.zeros(shape=dist_array_2.shape, dtype=np.float32))


def test_normalize_dict():
    dict_1 = {"hsv": np.array([1, 2, 3]), "embedding": np.array([2, 4, 6])}

    # deep copy dict
    dict_1_copy = {}
    for key, value in dict_1.items():
        dict_1_copy[key] = value.copy()

    norm_dict_1 = normalize_dict(dist_dict=dict_1)

    # check all keys are present
    assert "hsv" in norm_dict_1
    assert "embedding" in norm_dict_1

    # ensure normalization is correct
    assert np.allclose(norm_dict_1["hsv"], np.array([0.0, 0.5, 1.0], dtype=np.float32))
    assert np.allclose(norm_dict_1["embedding"], np.array([0.0, 0.5, 1.0], dtype=np.float32))

    dict_2 = {"hsv": np.array([1, 1, 1]), "embedding": np.array([2, 4, 6])}

    # deep copy dict
    dict_2_copy = {}
    for key, value in dict_2.items():
        dict_2_copy[key] = value.copy()

    norm_dict_2 = normalize_dict(dist_dict=dict_2)

    # check 0 array is returned for 0 variance dist arrays
    assert np.allclose(norm_dict_2["hsv"], np.array([0.0, 0.0, 0.0], dtype=np.float32))
    assert np.allclose(norm_dict_2["embedding"], np.array([0.0, 0.5, 1.0], dtype=np.float32))

    # ensure input dicts values aren't modified
    for key in dict_1:
        assert np.array_equal(dict_1[key], dict_1_copy[key])

    for key in dict_2:
        assert np.array_equal(dict_2[key], dict_2_copy[key])

    # check key set equality after normalization
    assert set(dict_1.keys()) == set(norm_dict_1.keys())
    assert set(dict_2.keys()) == set(norm_dict_2.keys())


def test_filter_0_variance():
    # check feature with 0 variance is removed
    norm_dict_1 = {"hsv": np.array([0.0, 0.0, 0.0]), "embedding": np.array([0.0, 0.5, 1.0])}
    filtered_dict_1 = filter_0_variance(norm_dist_dict=norm_dict_1)

    assert "hsv" not in filtered_dict_1
    assert np.allclose(filtered_dict_1["embedding"], np.array([0.0, 0.5, 1.0], dtype=np.float32))

    # check all features are removed if all have 0 variance
    norm_dict_2 = {"hsv": np.array([0.0, 0.0, 0.0]), "embedding": np.array([0.0, 0.0, 0.0])}
    filtered_dict_2 = filter_0_variance(norm_dist_dict=norm_dict_2)

    assert len(filtered_dict_2) == 0

    # check no features are removed if none have 0 variance
    norm_dict_3 = {"hsv": np.array([1.0, 0.5, 0.0]), "embedding": np.array([0.0, 0.5, 1.0])}
    filtered_dict_3 = filter_0_variance(norm_dist_dict=norm_dict_3)

    assert set(filtered_dict_3.keys()) == set(norm_dict_3.keys())
