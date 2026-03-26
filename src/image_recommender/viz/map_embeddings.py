import json
import logging
from pathlib import Path

import numpy as np

from image_recommender.config import DR_SEED
from image_recommender.features.storage import read_validate_shard
from image_recommender.viz.dr import compute_umap
from image_recommender.viz.plots import plot_2d, plot_3d


def run_map_embeddings(
    run_dir: Path,
    feature_type: str,
    dims: int,
    sample_size: int | None,
    output_dir: Path | None = None,
) -> None:
    """
    Maps embeddings to 2D or 3D space.

    Inputs:
    - run_dir (Where are the embedding shards stored?)
    - feature_type (Which feature type should be projected?)
    - dims (Target projection dimensionality: 2 or 3?)
    - sample_size (How many embeddings should be sampled before projection?)
    - output_dir (Where should the output be stored?)

    Outputs:
    - UMAP coordinates
    - metadata JSON
    - preview plot
    """
    embeddings_list = []
    ids_list = []

    shard_idx = 0

    # iterate sequential shard directories until a shard is missing

    while True:
        shard_path = run_dir / feature_type / f"shard_{shard_idx:04d}"

        if not shard_path.exists():
            break

        features, ids = read_validate_shard(
            run_dir=run_dir,
            feature_type=feature_type,
            shard_id=shard_idx,
            mmap=False,
        )

        embeddings_list.append(features)
        ids_list.extend(ids)

        shard_idx += 1

    if not embeddings_list:
        raise FileNotFoundError(f"No embedding shards found in {run_dir / feature_type}")

    embeddings = np.concatenate(embeddings_list, axis=0)

    # sampling
    if sample_size is not None and sample_size < len(ids_list):
        rng = np.random.default_rng(DR_SEED)
        idx = rng.choice(len(ids_list), size=sample_size, replace=False)
        embeddings = embeddings[idx]
        ids_list = [ids_list[i] for i in idx]

    logging.info(f"Loaded embeddings matrix with shape {embeddings.shape}")

    # UMAP
    logging.info("Running UMAP projection")

    n_neighbors = min(15, len(embeddings) - 1)

    coords = compute_umap(
        embeddings,
        n_components=dims,
        n_neighbors=n_neighbors,
    )

    logging.info(f"Generated coordinates with shape {coords.shape}")

    # output directory
    out_dir = output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # save coordinates
    coords_path = out_dir / f"coords_{dims}d.npy"
    ids_path = out_dir / f"coords_{dims}d_ids.npy"

    np.save(coords_path, coords)
    np.save(ids_path, np.array(ids_list, dtype=np.int32))

    logging.info(f"Saved coordinates: {coords_path}")

    # metadata
    metadata = {
        "algorithm": "umap",
        "feature_type": feature_type,
        "dims": dims,
        "seed": DR_SEED,
        "n_neighbors": n_neighbors,
        "sample_size": sample_size,
        "n_points": int(coords.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
    }

    meta_path = out_dir / f"coords_{dims}d_metadata.json"

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logging.info(f"Saved metadata: {meta_path}")

    # preview plot
    preview_name = f"preview_{dims}d.png"

    if dims == 2:
        plot_2d(coords, title="UMAP projection", run_dir=out_dir, filename=preview_name)
    else:
        plot_3d(coords, title="UMAP projection", run_dir=out_dir, filename=preview_name)

    logging.info("Preview plot generated")
