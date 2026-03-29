import numpy as np
import pytest

from image_recommender.recommender.scoring import validate_input


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
