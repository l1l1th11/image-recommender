import cv2
import numpy as np

from image_recommender.config import PHASH_DIM
from image_recommender.features.phash import extract_phash
from image_recommender.io.img_loader import load_rgb


def test_output():
    img = load_rgb("data/samples/image_1306.jpg")
    binary_vector = extract_phash(img)
    # resize
    img_resized = cv2.resize(img, (420, 67))
    binary_vector_resized = extract_phash(img_resized)

    # check output dimension, length, dtype and values
    vectors = [binary_vector, binary_vector_resized]

    for v in vectors:
        assert v.shape == (PHASH_DIM,)
        assert v.dtype == np.uint8
        assert np.all((v == 0) | (v == 1))


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
