import numpy as np

from image_recommender.metrics.hsv_distance import hsv_distance


def test_hsv_distance_self_distance():
    """Tests if self-distance is zero."""
    vec = np.random.rand(432).astype(np.float32)  # 432 = 12 * 6 * 6 bins
    assert hsv_distance(vec, vec) == 0.0  # Is the self-distance zero?


def test_hsv_distance_symmetry():
    """Tests symmetry between the vectors a and b."""
    a = np.random.rand(432).astype(np.float32)
    b = np.random.rand(432).astype(np.float32)
    np.testing.assert_almost_equal(
        hsv_distance(a, b), hsv_distance(b, a)
    )  # Is χ²(a, b) = χ²(b, a)?


def test_hsv_distance_pair():  # might seem redundant, but matches issue
    """Tests if self-distance is zero and smaller than distance to a different vector.
    Expectation: hsv_distance(a, a) = 0 and hsv_distance(a, b) > 0 for a != b.
    Similar to test_hsv_distance_self_distance, but this test focuses on a pair of different vectors.
    """
    a = np.zeros(432, dtype=np.float32)
    b = np.ones(432, dtype=np.float32)
    assert hsv_distance(a, a) < hsv_distance(
        a, b
    )  # Is the self-distance smaller than the distance to a different vector?
