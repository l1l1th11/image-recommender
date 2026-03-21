import numpy as np
import pytest

from image_recommender.viz.dr import compute_umap


def test_umap_shape_2d():
    """Tests that the output shape of compute_umap is correct in 2D."""
    X = np.random.rand(50, 128).astype(np.float32)

    coords = compute_umap(X, n_components=2)

    assert coords.shape == (50, 2)


def test_umap_shape_3d():
    """Tests that the output shape of compute_umap is correct in 3D."""
    X = np.random.rand(30, 64).astype(np.float32)

    coords = compute_umap(X, n_components=3)

    assert coords.shape == (30, 3)


def test_umap_deterministic():
    """Tests that the output of compute_umap is deterministic."""
    X = np.random.rand(100, 32).astype(np.float32)

    coords1 = compute_umap(X)
    coords2 = compute_umap(X)

    assert np.allclose(coords1, coords2)


def test_invalid_components():
    """Tests that ValueError is raised when n_components is not 2 or 3."""
    X = np.random.rand(10, 16).astype(np.float32)

    with pytest.raises(ValueError):
        compute_umap(X, n_components=4)


def test_invalid_shape():
    """Tests that ValueError is raised when the shape of X is invalid."""

    X = np.random.rand(128).astype(np.float32)

    with pytest.raises(ValueError):
        compute_umap(X)


def test_too_few_samples():
    """Tests that ValueError is raised when X has too few samples."""
    X = np.random.rand(1, 32).astype(np.float32)

    with pytest.raises(ValueError):
        compute_umap(X)


def test_protected_parameters():
    """Tests that ValueError is raised when protected parameters are overridden."""
    X = np.random.rand(20, 8).astype(np.float32)
    for param, val in [("metric", "euclidean"), ("random_state", 123)]:
        with pytest.raises(ValueError):
            compute_umap(X, **{param: val})


def test_n_neighbors_validation():
    """Tests that ValueError is raised when n_neighbors is invalid."""
    X = np.random.rand(10, 8).astype(np.float32)
    with pytest.raises(ValueError):
        compute_umap(X, n_neighbors=10)
    with pytest.raises(ValueError):
        compute_umap(X, n_neighbors=20)
