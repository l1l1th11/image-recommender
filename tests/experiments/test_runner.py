import json
import logging
from unittest.mock import patch

from experiments.runner import load_config, run_experiment

CONFIG_NAME = "pilot"


def test_pipeline_caching_and_umap_recompute(caplog, tmp_path):
    """Tests pipeline, caching and recomputation on UMAP parameter changes."""

    coords_dir = tmp_path / "coords" / CONFIG_NAME
    metadata_dir = tmp_path / "metadata" / CONFIG_NAME
    viz_dir = tmp_path / "viz" / CONFIG_NAME

    run_experiment(
        CONFIG_NAME,
        coords_dir=coords_dir,
        metadata_dir=metadata_dir,
        viz_dir=viz_dir,
    )

    coord_file = coords_dir / "coords_2d.npy"
    results_file = metadata_dir / "experiment_results.json"

    assert coord_file.exists(), "2D coordinates file was not created"
    assert results_file.exists(), "Experiment results file was not created"

    first_mtime = coord_file.stat().st_mtime

    caplog.clear()
    with caplog.at_level(logging.INFO):
        run_experiment(
            CONFIG_NAME,
            coords_dir=coords_dir,
            metadata_dir=metadata_dir,
            viz_dir=viz_dir,
        )

    log_messages = [record.message for record in caplog.records]
    assert any(
        "Using existing UMAP embeddings" in msg for msg in log_messages
    ), "Cached UMAP embeddings were not used on second run"

    second_mtime = coord_file.stat().st_mtime
    assert first_mtime == second_mtime, "UMAP embeddings file was recomputed instead of cached"

    original_config = load_config()
    modified_config = original_config.copy()
    modified_config[CONFIG_NAME]["umap"]["n_neighbors"] += 1  # force UMAP recomputation

    caplog.clear()
    with patch("experiments.runner.load_config", return_value=modified_config):
        with caplog.at_level(logging.INFO):
            run_experiment(
                CONFIG_NAME,
                coords_dir=coords_dir,
                metadata_dir=metadata_dir,
                viz_dir=viz_dir,
            )

    log_messages = [record.message for record in caplog.records]
    assert any(
        "Computing UMAP embeddings" in msg for msg in log_messages
    ), "UMAP was not recomputed after parameter change"

    with open(results_file) as f:
        results = json.load(f)

    assert "config" in results, "Experiment results missing 'config'"
    assert "outputs" in results, "Experiment results missing 'outputs'"
