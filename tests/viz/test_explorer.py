from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from image_recommender.viz.explorer import (
    create_thumbnail_cached,
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

    embeddings_root = tmp_path / "embeddings"
    shard_dir = embeddings_root / "shard_0001"
    shard_dir.mkdir(parents=True)

    embeddings = np.random.rand(4, 10)

    np.save(shard_dir / "features.npy", embeddings)
    np.save(shard_dir / "ids.npy", ids)

    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()

    for img_id in ids:
        Image.new("RGB", (32, 32), color=(100, 0, 0)).save(samples_dir / f"{img_id}.jpg")

    monkeypatch.setattr(
        "image_recommender.viz.explorer.get_path_by_id",
        lambda image_id, _: str(samples_dir / f"{image_id}.jpg"),
    )

    return coords_path, ids_path, embeddings_root, samples_dir


def _run_explorer(sample_data, return_fig=True):
    coords_path, ids_path, embeddings_root, samples_dir = sample_data

    return run_embedding_explorer(
        coords_path,
        ids_path,
        embeddings_path=embeddings_root,
        db_path=samples_dir,
        show=False,
        return_figure=return_fig,
    )


def test_load_coordinates(sample_data):
    """Tests that coordinates and IDs are loaded correctly."""
    coords_path, ids_path, _, _ = sample_data

    coords, ids = load_coordinates(coords_path, ids_path)

    assert coords.shape[1] == 2
    assert len(coords) == len(ids)


def test_invalid_paths():
    """Tests that invalid paths raise an error."""
    with pytest.raises(FileNotFoundError):
        load_coordinates(Path("invalid.npy"), Path("invalid.npy"))


def test_plot_renders(sample_data):
    """Tests that scatter plot is rendered correctly."""
    fig = _run_explorer(sample_data)

    assert fig is not None
    assert len(fig.data) > 0
    assert fig.data[0].type == "scattergl"  # Is it a ScatterGL plot?


def test_hover_thumbnail(sample_data):
    """Tests that hover interaction contains image information."""
    fig = _run_explorer(sample_data)

    hover_data = fig.data[0].text

    assert isinstance(hover_data, (list, tuple))
    assert len(hover_data) > 0
    assert all(isinstance(t, str) for t in hover_data)
    assert all("ID:" in t for t in hover_data)


def test_neighbor_grid(sample_data):
    """Tests that neighbor grid is rendered correctly."""
    coords_path, ids_path, _, samples_dir = sample_data

    _, ids = load_coordinates(coords_path, ids_path)

    grid = show_neighbor_grid(ids[:3].tolist(), samples_dir)

    assert isinstance(grid, list)
    assert len(grid) > 0


def test_missing_images(sample_data, monkeypatch):
    """Tests that missing images are handled correctly."""
    coords_path, ids_path, _, samples_dir = sample_data

    for f in samples_dir.iterdir():
        f.unlink()

    monkeypatch.setattr(
        "image_recommender.viz.explorer.get_path_by_id",
        lambda _, __: None,
    )

    _, ids = load_coordinates(coords_path, ids_path)

    grid = show_neighbor_grid(ids.tolist(), samples_dir)

    assert grid == []


def test_knn_uses_embeddings(sample_data):
    """Tests that KNN uses embeddings instead of UMAP."""
    coords_path, ids_path, embeddings_root, samples_dir = sample_data

    _, ids = load_coordinates(coords_path, ids_path)

    with patch("image_recommender.viz.explorer.NearestNeighbors") as MockNN:
        instance = MockNN.return_value

        run_embedding_explorer(
            coords_path,
            ids_path,
            embeddings_path=embeddings_root,
            db_path=samples_dir,
            show=False,
            return_figure=False,
        )

        instance.fit.assert_called()

        data = instance.fit.call_args[0][0]

        assert data.shape[1] > 2
        assert data.shape[0] == len(ids)


def test_thumbnail_cache(sample_data):
    """Tests that thumbnails are cached correctly."""
    _, _, _, samples_dir = sample_data

    path = str(next(iter(samples_dir.iterdir())))

    with patch("image_recommender.viz.explorer._create_thumbnail") as mock_create:
        mock_create.return_value = "cached"

        t1 = create_thumbnail_cached(path, 128)
        t2 = create_thumbnail_cached(path, 128)

        assert t1 == t2
        mock_create.assert_called_once()


def test_alignment_mismatch_between_coords_and_embeddings(sample_data):
    """Tests that mismatch between coords IDs and embedding IDs raises error."""
    coords_path, ids_path, embeddings_root, samples_dir = sample_data

    shard_ids_path = embeddings_root / "shard_0001" / "ids.npy"
    wrong_ids = np.array([999, 998, 997, 996])
    np.save(shard_ids_path, wrong_ids)

    with pytest.raises(ValueError):
        run_embedding_explorer(
            coords_path,
            ids_path,
            embeddings_path=embeddings_root,
            db_path=samples_dir,
            show=False,
        )
