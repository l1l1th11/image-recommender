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


def test_determinism():
    img = load_rgb("data/samples/image_1306.jpg")
    binary_vector_1 = extract_phash(img)
    binary_vector_2 = extract_phash(img)

    # check same image produces identical hashes
    assert np.array_equal(binary_vector_1, binary_vector_2)


def test_differing_hashes():
    img_1 = load_rgb("data/samples/image_1281.jpg")
    binary_vector_1 = extract_phash(img_1)
    img_2 = load_rgb("data/samples/image_0811.jpeg")
    binary_vector_2 = extract_phash(img_2)

    # check different images produce different hashes
    assert not np.array_equal(binary_vector_1, binary_vector_2)
