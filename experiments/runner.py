import json
from pathlib import Path

import numpy as np
import yaml

from image_recommender.config import DR_SEED
from image_recommender.viz.clustering import compute_kmeans
from image_recommender.viz.dr import compute_umap
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
    - coordinates (.npy) and IDs (.npy)
    - metadata (.json)
    - cluster labels (.npy)
    - preview plots (.png)
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

    coords_dict = {}

    embeddings, ids_list = run_map_embeddings(  # map embeddings pipeline
        run_dir=Path(cfg["run_dir"]),
        feature_type=cfg["feature_type"],
        dims=None,
        sample_size=cfg["sample_size"],
        umap_params=cfg["umap"],
        output_dir=None,
    )

    for dim in cfg["dims"]:
        coords = compute_umap(embeddings, n_components=dim, **cfg["umap"])
        coords_dict[dim] = coords

        coords_path = experiment_viz_dir / f"coords_{dim}d.npy"
        ids_path = experiment_viz_dir / f"coords_{dim}d_ids.npy"

        np.save(coords_path, coords)
        np.save(ids_path, np.array(ids_list, dtype=np.int32))

        meta = {
            "algorithm": "umap",
            "feature_type": cfg["feature_type"],
            "dims": dim,
            "seed": DR_SEED,
            "n_neighbors": min(cfg["umap"].get("n_neighbors", 15), embeddings.shape[0] - 1),
            "sample_size": cfg["sample_size"],
            "n_points": int(coords.shape[0]),
            "embedding_dim": int(embeddings.shape[1]),
        }

        meta_path = experiment_viz_dir / f"coords_{dim}d_metadata.json"

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    for dim, coords in coords_dict.items():
        preview_name = f"preview_{dim}d.png"

        if dim == 2:
            plot_2d(
                coords,
                point_size=cfg["point_size"],
                alpha=cfg["alpha"],
                title="UMAP projection",
                run_dir=experiment_viz_dir,
                filename=preview_name,
            )
        elif dim == 3:
            plot_3d(
                coords,
                point_size=cfg["point_size"],
                alpha=cfg["alpha"],
                title="UMAP projection",
                run_dir=experiment_viz_dir,
                filename=preview_name,
            )

    if 2 not in coords_dict:
        raise ValueError("Clustering requires 2D projection!")

    cluster_labels = compute_kmeans(embeddings, n_clusters=cfg["n_clusters"])

    for dim in cfg["dims"]:
        np.save(experiment_viz_dir / f"clusters_{dim}d.npy", cluster_labels)

    for dim, coords in coords_dict.items():
        filename = f"{dim}d_clusters.png"

        if dim == 2:
            plot_2d(
                coords,
                point_size=cfg["point_size"],
                alpha=cfg["alpha"],
                title=f"UMAP {dim}D clusters",
                run_dir=experiment_viz_dir,
                filename=filename,
                labels=cluster_labels,
            )
        elif dim == 3:
            plot_3d(
                coords,
                point_size=cfg["point_size"],
                alpha=cfg["alpha"],
                title=f"UMAP {dim}D clusters",
                run_dir=experiment_viz_dir,
                filename=filename,
                labels=cluster_labels,
            )

        results["outputs"].append(
            {
                "dims": dim,
                "coords_file": f"coords_{dim}d.npy",
                "clusters_file": f"clusters_{dim}d.npy",
                "plot_file": filename,
                "n_points": int(coords.shape[0]),
                "dimensionality": int(coords.shape[1]),
            }
        )

    with open(metadata_dir / "experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Experiment '{config_name}' completed.")
