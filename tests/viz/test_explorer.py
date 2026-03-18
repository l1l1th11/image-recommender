import json

import numpy as np
import pytest
from PIL import Image

from image_recommender.viz.explorer import (
    build_neighbor_model,
    create_thumbnail,
    load_coordinates,
    resolve_image_path,
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
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(ids.tolist(), f)

    samples_dir = tmp_path / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("image_recommender.viz.explorer.SAMPLES_DIR", samples_dir)

    for img_id in ids:
        img = Image.new("RGB", (32, 32), color=(img_id * 40, 0, 0))
        img.save(samples_dir / f"{img_id}.jpg")

    return coords_path, ids_path, ids


def test_load_coordinates(tmp_path):
    """Tests that coordinates and IDs are loaded correctly."""
    coords_path = tmp_path / "coords.npy"
    ids_path = tmp_path / "ids.npy"

    coords = np.random.rand(3, 2)
    ids = np.array([1, 2, 3])

    np.save(coords_path, coords)
    ids_path = tmp_path / "ids.json"
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(ids.tolist(), f)

    loaded_coords, loaded_ids = load_coordinates(coords_path, ids_path)
    assert np.array_equal(loaded_coords, coords)
    assert np.array_equal(loaded_ids, ids)


def test_run_embedding_explorer(tmp_path):
    """Tests that a figure is returned."""
    coords_path = tmp_path / "coords.npy"
    ids_path = tmp_path / "ids.npy"

    coords = np.random.rand(3, 2)
    ids = np.array([1, 2, 3])

    np.save(coords_path, coords)
    ids_path = tmp_path / "ids.json"
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(ids.tolist(), f)

    fig = run_embedding_explorer(coords_path, ids_path, show=False, return_figure=True)
    assert fig is not None
    assert len(fig.data) == 1

    hover = fig.data[0].text
    assert all("ID:" in h for h in hover)


def test_invalid_coordinate_paths(tmp_path):
    """Tests that invalid coordinate paths raise an error."""
    coords_path = tmp_path / "missing_coords.npy"
    ids_path = tmp_path / "missing_ids.npy"
    with pytest.raises(FileNotFoundError):
        load_coordinates(coords_path, ids_path)


def test_resolve_image_path():
    """Tests that missing images return None."""
    result = resolve_image_path(999999)  # Non-existent ID

    assert result is None  # Image not found


def test_thumbnail_generation(tmp_path):
    """Tests that thumbnails are created."""
    img_path = tmp_path / "img.jpg"

    img = Image.new("RGB", (32, 32), color=(255, 0, 0))
    img.save(img_path)

    thumb = create_thumbnail(img_path)

    assert thumb.startswith("data:image/png;base64,")


def test_build_neighbor_model(sample_data):
    """Tests that neighbor model builds correctly."""
    coords_path, _, _ = sample_data
    coords = np.load(coords_path)

    nn = build_neighbor_model(coords)
    dists, inds = nn.kneighbors([coords[0]])

    expected_neighbors = min(9, len(coords))
    assert inds.shape == (1, expected_neighbors)
    assert len(dists[0]) == expected_neighbors


def test_show_neighbor_grid(sample_data):
    """Tests that neighbor grid runs without error."""
    _, _, ids = sample_data

    show_neighbor_grid(ids[:3], show=False)
