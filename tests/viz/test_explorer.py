import numpy as np

from image_recommender.viz.explorer import load_coordinates, run_embedding_explorer


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
    assert all(str(i) in fig.data[0].text for i in ids)
