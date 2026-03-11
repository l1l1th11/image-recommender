import numpy as np

from image_recommender.config import PHASH_DIM
from image_recommender.features.phash import extract_phash
from image_recommender.io.img_loader import load_rgb


def test_output():
    img = load_rgb("data/samples/image_1306.jpg")
    binary_vector = extract_phash(img)

    # check output dimension, length, dtype and values
    assert binary_vector.shape == (PHASH_DIM,)
    assert binary_vector.dtype == np.uint8
    assert np.all((binary_vector == 0) | (binary_vector == 1))
