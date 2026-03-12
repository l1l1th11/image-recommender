import numpy as np
import pytest

from image_recommender.viz.plots import plot_2d, plot_3d, validate_coordinates


def test_plot_functions_do_not_modify_input():
    """Tests that plotting functions do not modify input coordinate arrays."""
    coords_2d = np.random.rand(5, 2)
    coords_3d = np.random.rand(5, 3)

    coords_2d_copy = coords_2d.copy()
    coords_3d_copy = coords_3d.copy()

    plot_2d(coords_2d)
    plot_3d(coords_3d)

    np.testing.assert_array_equal(coords_2d, coords_2d_copy)
    np.testing.assert_array_equal(coords_3d, coords_3d_copy)


def test_plot_2d_creates_figure(tmp_path):
    """Tests that plot_2d creates a figure and saves it to a file."""
    coords = np.random.rand(5, 2)
    fig = plot_2d(coords, run_dir=tmp_path)
    output_file = tmp_path / "plot_2d.png"
    assert fig is not None
    assert output_file.exists()


def test_plot_3d_creates_figure(tmp_path):
    """Tests that plot_3d creates a figure and saves it to a file."""
    coords = np.random.rand(5, 3)
    fig = plot_3d(coords, run_dir=tmp_path)
    output_file = tmp_path / "plot_3d.png"
    assert fig is not None
    assert output_file.exists()


def test_invalid_coordinates_raises():
    """Tests that ValueError is raised when coordinates are invalid."""
    coords_invalid = np.random.rand(5, 4)

    with pytest.raises(ValueError):
        validate_coordinates(coords_invalid, expected_dim=2)

    with pytest.raises(ValueError):
        validate_coordinates(coords_invalid, expected_dim=3)
