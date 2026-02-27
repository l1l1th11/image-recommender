import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from image_recommender.features.embedding import extract_embeddings_batch
from image_recommender.features.samples_driver_embedding import (
    compute_topk,
    load_sample_images,
)


def _create_test_image(path: Path, color: tuple[int, int, int]) -> None:
    """
    Creates a deterministic RGB image with a single solid color.
    """
    img = Image.new("RGB", (32, 32), color=color)
    img.save(path)


def test_samples_knn_end_to_end(tmp_path: Path):
    """
    Tests the end-to-end functionality of loading sample images,
    extracting embeddings and computing top-k nearest neighbors.
    """

    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()

    _create_test_image(samples_dir / "red.png", (255, 0, 0))
    _create_test_image(samples_dir / "green.png", (0, 255, 0))
    _create_test_image(samples_dir / "blue.png", (0, 0, 255))

    ids, images = load_sample_images(samples_dir)

    assert len(ids) == 3  # Is the correct number of images loaded?

    embeddings = extract_embeddings_batch(
        images,
        model_name="resnet18",
        device="cpu",
        pretrained=False,
    )

    k = 2  # Set top-k neighbors, query itself will always be top-1

    results = compute_topk(ids, embeddings, k=k)

    assert len(results) == 3  # Do we have results for each query?

    for i, neighbors in enumerate(results):
        assert len(neighbors) == k  # Do we have k neighbors for each query?

        top_id, top_dist = neighbors[0]

        assert top_id == ids[i]  # Is the closest neighbor the query itself?

        assert np.isclose(top_dist, 0.0, atol=1e-6)  # Is the distance to self approximately zero?


def test_cli_embedding_on_samples_runs(tmp_path):
    """
    Tests that the "embedding-on-samples" CLI command runs successfully on sample images.
    """
    samples_dir = tmp_path / "data" / "samples"
    samples_dir.mkdir(parents=True)
    _create_test_image(samples_dir / "red.png", (255, 0, 0))
    _create_test_image(samples_dir / "green.png", (0, 255, 0))

    src_path = Path(__file__).parents[2] / "src"
    env = {**os.environ, "PYTHONPATH": str(src_path)}

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "image_recommender.cli.main",
            "embedding-on-samples",
            "--k",
            "2",
            "--model",
            "resnet18",
            "--device",
            "cpu",
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, f"CLI failed:\n{result.stderr}"  # Did the CLI command succeed?

    assert "Query:" in result.stdout  # Does the output contain expected query information?
    assert (
        ".png" in result.stdout or ".jpg" in result.stdout
    )  # Does the output mention image files?
