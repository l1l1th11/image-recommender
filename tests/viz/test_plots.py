import numpy as np
import pytest

from image_recommender.viz.plots import plot_2d, plot_3d, validate_coordinates


def test_plot_functions_exist():
    """Tests that the plot functions exist."""
    coords_2d = np.random.rand(5, 2)
    coords_3d = np.random.rand(5, 3)

    fig2d = plot_2d(coords_2d)
    fig3d = plot_3d(coords_3d)

    assert fig2d is not None
    assert fig3d is not None


def test_invalid_coordinates_raises():
    """Tests that ValueError is raised when coordinates are invalid."""
    coords_invalid = np.random.rand(5, 4)

    with pytest.raises(ValueError):
        validate_coordinates(coords_invalid, expected_dim=2)

    with pytest.raises(ValueError):
        validate_coordinates(coords_invalid, expected_dim=3)
