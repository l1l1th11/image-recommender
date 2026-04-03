import json
import logging
from unittest.mock import patch

import numpy as np

from experiments.runner import load_config, run_experiment

CONFIG_NAME = "pilot"


def test_pipeline_caching_and_umap_recompute(caplog, tmp_path):
    """Tests pipeline, caching, UMAP recomputation, reproducibility and outputs."""

    coords_dir = tmp_path / "coords" / CONFIG_NAME
    metadata_dir = tmp_path / "metadata" / CONFIG_NAME
    viz_dir = tmp_path / "viz" / CONFIG_NAME

    run_experiment(  # 1st run: Tests that UMAP embeddings are computed and outputs are created.
        CONFIG_NAME,
        coords_dir=coords_dir,
        metadata_dir=metadata_dir,
        viz_dir=viz_dir,
    )

    coord_file_2d = coords_dir / "coords_2d.npy"
    coord_file_3d = coords_dir / "coords_3d.npy"
    results_file = metadata_dir / "experiment_results.json"
    clusters_2d_file = viz_dir / "clusters_2d.npy"
    clusters_3d_file = viz_dir / "clusters_3d.npy"
    preview_2d_file = viz_dir / "preview_2d.png"
    preview_3d_file = viz_dir / "preview_3d.png"

    for file in [
        coord_file_2d,
        coord_file_3d,
        results_file,
        clusters_2d_file,
        clusters_3d_file,
        preview_2d_file,
        preview_3d_file,
    ]:
        assert file.exists(), f"{file} was not created"  # Do the expected output files exist?

    first_mtime = (
        coord_file_2d.stat().st_mtime
    )  # timestamp of the 2D coordinates file after the first run

    caplog.clear()
    with caplog.at_level(logging.INFO):
        run_experiment(  # 2nd run: Tests that cached UMAP embeddings are used and outputs are not recomputed.
            CONFIG_NAME,
            coords_dir=coords_dir,
            metadata_dir=metadata_dir,
            viz_dir=viz_dir,
        )

    log_messages = [record.message for record in caplog.records]
    assert any(
        "Using existing UMAP embeddings" in msg for msg in log_messages
    ), "Cached UMAP embeddings were not used on second run"

    assert (
        first_mtime == coord_file_2d.stat().st_mtime
    ), "UMAP embeddings file was recomputed instead of cached"

    coords_2d = np.load(coord_file_2d)
    coords_3d = np.load(coord_file_3d)
    clusters_2d = np.load(clusters_2d_file)
    clusters_3d = np.load(clusters_3d_file)
    ids = np.load(coords_dir / "coords_2d_ids.npy")

    # Are the clusters distinct?
    assert len(np.unique(clusters_2d)) > 1
    assert len(np.unique(clusters_3d)) > 1

    # Do the coordinates have variance?
    assert (coords_2d.std(axis=0) > 0).all()
    assert (coords_3d.std(axis=0) > 0).all()

    # Is the number of coordinates, clusters and ids consistent?
    assert len(coords_2d) == len(clusters_2d) == len(ids)
    assert len(coords_3d) == len(clusters_3d) == len(ids)

    # Does the results file contain expected keys and values?
    with open(results_file) as f:
        results = json.load(f)
    assert "config" in results
    assert "outputs" in results

    for output in results["outputs"]:
        for k in ["coords_file", "clusters_file", "plot_file", "n_points", "dimensionality"]:
            assert k in output
        assert output["n_points"] > 0
        assert output["dimensionality"] in [2, 3]

    modified_config = load_config().copy()
    modified_config[CONFIG_NAME]["umap"]["n_neighbors"] += 1  # force UMAP recomputation

    caplog.clear()
    with (
        patch("experiments.runner.load_config", return_value=modified_config),
        caplog.at_level(logging.INFO),
    ):
        run_experiment(  # 3rd run: Tests that UMAP is recomputed after parameter change.
            CONFIG_NAME,
            coords_dir=coords_dir,
            metadata_dir=metadata_dir,
            viz_dir=viz_dir,
        )

    log_messages = [record.message for record in caplog.records]
    assert any(
        "Computing UMAP embeddings" in msg for msg in log_messages
    ), "UMAP was not recomputed after parameter change"

    alt_coords_dir = tmp_path / "coords" / f"{CONFIG_NAME}_alt_seed"
    alt_metadata_dir = tmp_path / "metadata" / f"{CONFIG_NAME}_alt_seed"
    alt_viz_dir = tmp_path / "viz" / f"{CONFIG_NAME}_alt_seed"

    with patch("image_recommender.viz.dr.DR_SEED", 43):
        run_experiment(  # 4th run: Tests that different seed produces different coordinates.
            CONFIG_NAME,
            coords_dir=alt_coords_dir,
            metadata_dir=alt_metadata_dir,
            viz_dir=alt_viz_dir,
        )

    coords_alt = np.load(alt_coords_dir / "coords_2d.npy")
    assert not np.array_equal(
        coords_2d, coords_alt
    ), "Different seed did not produce different coordinates"
