from pathlib import Path

import numpy as np
import yaml

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

    experiment_viz_dir = Path("data/experiments/viz") / config_name  # output directory

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

        if dims == 2:
            plot_2d(
                coords,
                title=f"UMAP {dims}D clusters",
                run_dir=experiment_viz_dir,
                filename=f"{dims}d_clusters.png",
                labels=cluster_labels,
            )
        elif dims == 3:
            plot_3d(
                coords,
                title=f"UMAP {dims}D clusters",
                run_dir=experiment_viz_dir,
                filename=f"{dims}d_clusters.png",
                labels=cluster_labels,
            )

    print(f"Experiment '{config_name}' completed.")
