import numpy as np

from image_recommender.features.hsv import hsv_features


def test_hsv_features_shape():
    """Checks that the output has the expected fixed dimension."""
    img = np.random.randint(0, 256, size=(10, 10, 3), dtype=np.uint8)
    feat = hsv_features(img)
    assert isinstance(feat, np.ndarray)
    assert feat.ndim == 1  # Is the output a 1D vector?
    assert feat.size > 0  # Is the output non-empty?


def test_hsv_features_idempotent():
    """Same input should produce exactly the same output."""
    img = np.random.randint(0, 256, size=(5, 5, 3), dtype=np.uint8)
    feat1 = hsv_features(img)
    feat2 = hsv_features(img)
    np.testing.assert_array_equal(feat1, feat2)  # Are the outputs the same?


def test_hsv_features_values_range():
    """Checks that the values are non-negative and normalized."""
    img = np.random.randint(0, 256, size=(8, 8, 3), dtype=np.uint8)
    feat = hsv_features(img)
    assert np.all(feat >= 0)  # Are values between 0...
    assert np.all(feat <= 1)  # ... and 1?
    np.testing.assert_almost_equal(feat.sum(), 1.0, decimal=5)
