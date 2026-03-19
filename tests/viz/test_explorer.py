from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from image_recommender.viz.explorer import (
    load_coordinates,
    run_embedding_explorer,
    show_neighbor_grid,
)


@pytest.fixture
def sample_data(tmp_path, monkeypatch):
    """Creates sample data for testing."""
    coords = np.random.rand(4, 2)
    ids = np.array([1, 2, 3, 4])

    coords_path = tmp_path / "coords.npy"
    ids_path = tmp_path / "ids.npy"

    np.save(coords_path, coords)
    np.save(ids_path, ids)

    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()

    monkeypatch.setattr(
        "image_recommender.viz.explorer.SAMPLES_DIR",
        samples_dir,
    )

    for img_id in ids:
        img = Image.new("RGB", (32, 32), color=(100, 0, 0))
        img.save(samples_dir / f"{img_id}.jpg")

    return coords_path, ids_path, samples_dir


def test_load_coordinates(sample_data):
    """Tests that coordinates and IDs are loaded correctly."""
    coords_path, ids_path, _ = sample_data

    coords, ids = load_coordinates(coords_path, ids_path)

    assert coords.shape[1] == 2
    assert len(coords) == len(ids)


def test_plot_renders(sample_data):
    """Tests that scatter plot is rendered correctly."""
    coords_path, ids_path, samples_dir = sample_data

    fig = run_embedding_explorer(
        coords_path,
        ids_path,
        db_path=samples_dir,
        show=False,
        return_figure=True,
    )

    assert fig is not None
    assert len(fig.data) > 0
    assert fig.data[0].type == "scattergl"  # Is it a ScatterGL plot?


def test_hover_thumbnail(sample_data):
    """Tests that hover interaction contains image information."""
    coords_path, ids_path, samples_dir = sample_data

    fig = run_embedding_explorer(
        coords_path,
        ids_path,
        db_path=samples_dir,
        show=False,
        return_figure=True,
    )

    hover_data = fig.data[0].text

    assert hover_data is not None
    assert all("ID:" in text for text in hover_data)


def test_click_neighbor_grid(sample_data):
    """Tests that neighbor grid is rendered correctly."""
    coords_path, ids_path, samples_dir = sample_data

    _, ids = load_coordinates(coords_path, ids_path)

    neighbor_ids = ids[:3]

    grid = show_neighbor_grid(neighbor_ids.tolist(), samples_dir)

    assert isinstance(grid, list)
    assert len(grid) > 0


def test_invalid_coordinate_paths():
    """Tests that invalid coordinate paths raise an error."""
    with pytest.raises(FileNotFoundError):
        load_coordinates(
            Path("does_not_exist.npy"),
            Path("does_not_exist.npy"),
        )


def test_missing_images(sample_data, monkeypatch):
    """Tests that missing images are handled correctly."""
    coords_path, ids_path, samples_dir = sample_data

    for file in samples_dir.iterdir():
        file.unlink()

    monkeypatch.setattr(
        "image_recommender.viz.explorer.SAMPLES_DIR",
        samples_dir,
    )

    _, ids = load_coordinates(coords_path, ids_path)

    grid = show_neighbor_grid(ids.tolist(), samples_dir)

    assert isinstance(grid, list)
    assert len(grid) == 0
