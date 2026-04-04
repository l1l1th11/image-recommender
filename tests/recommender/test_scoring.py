import numpy as np
import pytest

from image_recommender.recommender.scoring import (
    compute_scores,
    filter_0_variance,
    normalize_array,
    normalize_dict,
    score_candidates,
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


def test_score_candidates():
    # test basic fusion
    filtered_dict_1 = {"hsv": np.array([1.0, 0.5, 0.0]), "embedding": np.array([0.0, 0.5, 0.2])}
    score_arr_1 = score_candidates(filtered_dist_dict=filtered_dict_1)
    assert np.allclose(score_arr_1, np.array([0.5, 0.5, 0.1], dtype=np.float32))

    # test single feature
    filtered_dict_2 = {"hsv": np.array([0.0, 0.5, 1.0])}
    score_arr_2 = score_candidates(filtered_dist_dict=filtered_dict_2)
    assert np.allclose(score_arr_2, np.array([0.0, 0.5, 1.0], dtype=np.float32))

    # check shape correctness
    assert score_arr_1.shape == (3,)
    assert score_arr_2.shape == (3,)


def test_compute_scores():
    # check basic fusion
    dict_1 = {"hsv": np.array([1, 5, 7]), "embedding": np.array([4, 9, 3])}
    weights_1 = {"hsv": 0.3, "embedding": 0.7}
    score_arr_1 = compute_scores(dist_dict=dict_1, weights=weights_1)
    assert np.allclose(score_arr_1, np.array([0.1166666667, 0.9, 0.3], dtype=np.float32))

    # test renormalization after filtering
    dict_2 = {
        "hsv": np.array([1, 2, 7]),
        "embedding": np.array([5, 9, 6]),
        "phash": np.array([2, 2, 2]),
    }
    weights_2 = {"hsv": 0.2, "embedding": 0.6, "phash": 0.2}
    score_arr_2 = compute_scores(dist_dict=dict_2, weights=weights_2)
    assert np.allclose(score_arr_2, np.array([0.0, 0.7916666667, 0.4375], dtype=np.float32))

    # check empty dict after filtering leads to 0 array
    dict_3 = {"hsv": np.array([1, 1, 1]), "embedding": np.array([2, 2, 2])}
    score_arr_3 = compute_scores(dist_dict=dict_3)
    assert np.allclose(score_arr_3, np.array([0.0, 0.0, 0.0], dtype=np.float32))

    # test invalid weights
    dict_4 = {"hsv": np.array([1, 2, 3]), "embedding": np.array([4, 5, 6])}

    # wrong keys
    weights_3 = {"phash": 0.3, "embedding": 0.7}
    with pytest.raises(ValueError):
        compute_scores(dist_dict=dict_4, weights=weights_3)

    # sum != 1
    weights_4 = {"hsv": 0.7, "embedding": 0.7}
    with pytest.raises(ValueError):
        compute_scores(dist_dict=dict_4, weights=weights_4)

    # negative weight
    weights_5 = {"hsv": -0.3, "embedding": 0.7}
    with pytest.raises(ValueError):
        compute_scores(dist_dict=dict_4, weights=weights_5)
