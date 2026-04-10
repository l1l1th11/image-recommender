import json
from pathlib import Path

import numpy as np

from image_recommender.cli.main import main


def create_dummy_shards(run_dir: Path, feature_type: str, n_shards=2, n_points=5, dim=4):
    """Creates dummy shards for testing."""
    from image_recommender.features.storage import (
        mark_success,
        write_validate_shard_atomic,
    )

    rng = np.random.default_rng(0)
    for shard_id in range(n_shards):
        features = rng.random((n_points, dim), dtype=np.float32)
        ids = list(range(shard_id * n_points, (shard_id + 1) * n_points))
        meta = {
            "feature_type": feature_type,
            "feature_dim": dim,
            "feature_dtype": str(features.dtype),
            "shard_size": n_points,
            "created_at": "2026-03-13T00:00:00",
            "version": 1,
        }
        write_validate_shard_atomic(
            run_dir=run_dir,
            shard_id=shard_id,
            feature_type=feature_type,
            features=features,
            ids=ids,
            meta=meta,
        )
        mark_success(run_dir=run_dir, feature_type=feature_type, shard_id=shard_id)


def test_cli_runs_success(tmp_path):
    """Tests that CLI runs successfully."""
    run_dir = tmp_path / "run"
    feature_type = "embedding"
    create_dummy_shards(run_dir=run_dir, feature_type=feature_type, n_shards=1, n_points=5, dim=4)

    # CLI arguments
    argv = [
        "map-embeddings",
        "--run-dir",
        str(run_dir),
        "--feature-type",
        feature_type,
        "--dims",
        "2",
        "--sample-size",
        "5",
    ]

    # Run CLI
    ret_code = main(argv)
    assert ret_code == 0, "CLI failed unexpectedly!"

    # Check output files
    viz_dir = run_dir / feature_type / "viz"
    coords_file = viz_dir / "coords_2d.npy"
    meta_file = viz_dir / "coords_2d_metadata.json"
    # preview_file = viz_dir / "preview_2d.png"

    assert coords_file.exists(), "NumPy coordinates file missing!"
    assert meta_file.exists(), "Metadata JSON file missing!"
    # assert preview_file.exists(), "Preview PNG plot missing!"
    # assert preview_file.stat().st_size > 0, "Preview PNG file is empty!"

    # Check coordinates shape
    coords = np.load(coords_file)
    assert coords.shape[0] == 5, "Number of coordinates does not match sample-size!"
    assert coords.shape[1] == 2, "Coordinates dimensionality mismatch!"

    # Check metadata
    with open(meta_file) as f:
        meta = json.load(f)
    assert meta["dims"] == 2, "Metadata dims mismatch!"
    assert meta["algorithm"] == "umap", "Metadata algorithm mismatch!"


def test_cli_runs_success_3d(tmp_path):
    """Tests that CLI runs successfully on pilot embedding dataset in 3D."""
    run_dir = tmp_path / "run3d"
    feature_type = "embedding"
    create_dummy_shards(run_dir=run_dir, feature_type=feature_type, n_shards=1, n_points=5, dim=4)

    argv = [
        "map-embeddings",
        "--run-dir",
        str(run_dir),
        "--feature-type",
        feature_type,
        "--dims",
        "3",
        "--sample-size",
        "5",
    ]

    ret_code = main(argv)
    assert ret_code == 0, "Shards should be loaded successfully!"

    coords_file = run_dir / feature_type / "viz" / "coords_3d.npy"
    assert coords_file.exists()
    coords = np.load(coords_file)
    assert coords.shape[1] == 3


def test_map_embeddings_no_shards(tmp_path):
    """Tests that the function fails when no shards are present."""
    run_dir = tmp_path / "empty_run"
    feature_type = "embedding"

    argv = [
        "map-embeddings",
        "--run-dir",
        str(run_dir),
        "--feature-type",
        feature_type,
        "--dims",
        "2",
    ]

    ret_code = main(argv)
    assert ret_code == 1, "Missing shards should trigger an error!"
