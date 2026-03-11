import numpy as np

from image_recommender.metrics.hamming import hamming_distance


def test_self_distance():
    vector = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
    result = hamming_distance(query=vector, candidate=vector)

    assert result == 0
