import os

import numpy as np
import pytest

from image_recommender.features.embedding import (
    extract_embedding,
    extract_embeddings_batch,
    get_embedding_dim,
)


def _dummy_img(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)


def test_single_embedding_shape_and_dtype():
    img = _dummy_img()
    emb = extract_embedding(img, pretrained=False)
    assert emb.ndim == 1  # Is the embedding one-dimensional?
    assert emb.dtype == np.float32  # Is the embedding of type float32?
    assert emb.shape[0] == get_embedding_dim(
        "resnet18"
    )  # Does the embedding have the correct dimension?


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Skip heavy pretrained test in CI")
def test_local_smoke_pretrained_true():
    img = _dummy_img()
    emb = extract_embedding(img, pretrained=True)
    assert emb.shape[0] == get_embedding_dim("resnet18")
    assert emb.dtype == np.float32


def test_batch_embedding_shape_and_consistency():
    imgs = [_dummy_img(1), _dummy_img(2), _dummy_img(3)]
    singles = [extract_embedding(img, pretrained=False) for img in imgs]
    batch = extract_embeddings_batch(imgs, pretrained=False, batch_size=2)
    assert batch.shape == (3, singles[0].shape[0])  # Has the embedding correct shape?
    for i in range(3):
        np.testing.assert_allclose(
            batch[i], singles[i], rtol=1e-5, atol=1e-6
        )  # Are embeddings within tolerance?


def test_determinism_pretrained_false():
    img = _dummy_img(42)
    emb1 = extract_embedding(img, pretrained=False)
    emb2 = extract_embedding(img, pretrained=False)
    np.testing.assert_array_equal(emb1, emb2)  # Are embeddings (for same seed) exactly equal?


def test_embedding_changes_with_input():
    img1 = np.zeros((224, 224, 3), dtype=np.uint8)
    img2 = np.ones((224, 224, 3), dtype=np.uint8) * 255
    emb1 = extract_embedding(img1, pretrained=False)
    emb2 = extract_embedding(img2, pretrained=False)
    assert not np.allclose(emb1, emb2)  # Are embeddings different for distinct inputs?
