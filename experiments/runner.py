from pathlib import Path

import yaml

from image_recommender.viz.map_embeddings import run_map_embeddings


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
        run_map_embeddings(  # map embeddings pipeline
            run_dir=Path(cfg["run_dir"]),
            feature_type=cfg["feature_type"],
            dims=dims,
            sample_size=cfg["sample_size"],
            output_dir=experiment_viz_dir,
        )

    print(f"Experiment '{config_name}' completed.")
