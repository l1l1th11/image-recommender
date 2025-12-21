import os

import numpy as np
import pytest

from image_recommender.features.embedding import extract_embedding, get_embedding_dim


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
