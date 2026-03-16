import numpy as np
from PIL import Image

from image_recommender.viz.explorer import (
    build_hover_data,
    create_thumbnail,
    load_coordinates,
    resolve_image_path,
    run_embedding_explorer,
)


def test_load_coordinates(tmp_path):
    """Tests that coordinates and IDs are loaded correctly."""
    coords_path = tmp_path / "coords.npy"
    ids_path = tmp_path / "ids.npy"

    coords = np.random.rand(3, 2)
    ids = np.array([1, 2, 3])

    np.save(coords_path, coords)
    np.save(ids_path, ids)

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
    np.save(ids_path, ids)

    fig = run_embedding_explorer(coords_path, ids_path, show=False, return_figure=True)
    assert fig is not None
    assert len(fig.data) == 1

    hover = fig.data[0].text
    assert all("ID:" in h for h in hover)
    assert any("<img" in h for h in hover)


def test_hover_data_generation():
    """Tests that hover text is generated for each ID."""
    ids = np.array([1, 2, 3])

    hover = build_hover_data(ids)

    assert len(hover) == 3
    assert "<img" in hover[0]
    assert "ID:" in hover[0]


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
