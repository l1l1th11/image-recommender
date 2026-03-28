import json
from pathlib import Path

import numpy as np
import yaml

from image_recommender.config import DR_SEED
from image_recommender.viz.clustering import compute_kmeans
from image_recommender.viz.map_embeddings import run_map_embeddings
from image_recommender.viz.plots import plot_2d, plot_3d


def load_config() -> dict:
    """
    Loads experiment configuration.
    """
    config_path = Path("experiments/params.yaml")

    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_experiment(config_name: str) -> None:
    """
    Runs the experiment.
    Input: config_name (name of experiment configuration)
    Output:
    - 2D coordinates (.npy) and IDs (.npy)
    - 2D metadata (.json)
    - 3D coordinates (.npy) and IDs (.npy)
    - 3D metadata (.json) and preview plots (.png)
    """
    config = load_config()

    cfg = config[config_name]

    # output directories
    experiment_viz_dir = Path("data/experiments/viz") / config_name
    experiment_viz_dir.mkdir(parents=True, exist_ok=True)

    metadata_dir = Path("data/experiments/metadata") / config_name
    metadata_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "config": config_name,
        "run_dir": cfg["run_dir"],
        "feature_type": cfg["feature_type"],
        "sample_size": cfg["sample_size"],
        "dims": cfg["dims"],
        "n_clusters": cfg["n_clusters"],
        "seed": DR_SEED,
        "outputs": [],
    }

    for dims in cfg["dims"]:  # 2D and 3D
        coords = run_map_embeddings(  # map embeddings pipeline
            run_dir=Path(cfg["run_dir"]),
            feature_type=cfg["feature_type"],
            dims=dims,
            sample_size=cfg["sample_size"],
            output_dir=experiment_viz_dir,
        )

        cluster_labels = compute_kmeans(coords, n_clusters=cfg["n_clusters"])

        np.save(experiment_viz_dir / f"clusters_{dims}d.npy", cluster_labels)

        plot_filename = f"{dims}d_clusters.png"

        if dims == 2:
            plot_2d(
                coords,
                title=f"UMAP {dims}D clusters",
                run_dir=experiment_viz_dir,
                filename=plot_filename,
                labels=cluster_labels,
            )
        elif dims == 3:
            plot_3d(
                coords,
                title=f"UMAP {dims}D clusters",
                run_dir=experiment_viz_dir,
                filename=plot_filename,
                labels=cluster_labels,
            )

        results["outputs"].append(
            {
                "dims": dims,
                "coords_file": f"coords_{dims}d.npy",
                "clusters_file": f"clusters_{dims}d.npy",
                "plot_file": plot_filename,
                "n_points": int(coords.shape[0]),
                "dimensionality": int(coords.shape[1]),
            }
        )

    with open(metadata_dir / "experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Experiment '{config_name}' completed.")
