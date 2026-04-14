from pathlib import Path

import numpy as np

from image_recommender.constants import SUPPORTED_FEATURES
from image_recommender.features.storage import read_validate_shard


def build_feature_mapping(run_dir: Path, feature: str):
    feature_dir = run_dir / feature

    shard_dirs = sorted(p for p in feature_dir.glob("shard_*") if p.is_dir())

    all_ids = []
    all_vecs = []

    for shard_dir in shard_dirs:
        shard_id = int(shard_dir.name.split("_")[1])

        features, ids = read_validate_shard(
            run_dir=run_dir,
            feature_type=feature,
            shard_id=shard_id,
        )

        all_ids.append(ids)
        all_vecs.append(features)

    ids = np.concatenate(all_ids)
    vecs = np.vstack(all_vecs)

    out_dir = feature_dir / "mapping"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "ids.npy", ids)
    np.save(out_dir / "vecs.npy", vecs)

    print(f"{feature}: saved {len(ids)} vectors")


def main():
    run_dir = Path("data/features/db")

    for feature in SUPPORTED_FEATURES:
        print(f"\nBuilding mapping for {feature}...")
        build_feature_mapping(run_dir, feature)


if __name__ == "__main__":
    main()
