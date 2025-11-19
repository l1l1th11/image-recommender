import numpy as np

from image_recommender.features.hsv import hsv_features

H_BINS = 12
S_BINS = 6
V_BINS = 6
EXPECTED_SIZE = H_BINS * S_BINS * V_BINS  # --> 12 * 6 * 6 = 432


def test_hsv_features_shape():
    """Checks that the output has the expected fixed dimension and valid dtype/values."""
    img = np.random.randint(0, 256, size=(10, 10, 3), dtype=np.uint8)
    feat = hsv_features(img)
    assert isinstance(feat, np.ndarray)  # Is the output a numpy array?
    assert feat.ndim == 1  # Is it 1D?
    assert feat.size == EXPECTED_SIZE  # Is the size correct?
    assert feat.dtype == np.float32  # Is the dtype correct?
    assert np.all(feat >= 0) and np.all(feat <= 1)  # Values between 0 and 1?


def test_hsv_features_idempotent():
    """Same input should produce exactly the same output."""
    img = np.random.randint(0, 256, size=(5, 5, 3), dtype=np.uint8)
    feat1 = hsv_features(img)
    feat2 = hsv_features(img)
    np.testing.assert_array_equal(feat1, feat2)


def test_hsv_features_normalization():
    """Checks that the values are normalized to sum 1."""
    img = np.random.randint(0, 256, size=(8, 8, 3), dtype=np.uint8)
    feat = hsv_features(img)
    np.testing.assert_almost_equal(feat.sum(), 1.0, decimal=5)  # Is the sum approximately 1?
