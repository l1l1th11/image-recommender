from pathlib import Path

import numpy as np

from image_recommender.features.storage import read_validate_shard


def load_canonical_ids(run_dir: Path | str, feature_type: str) -> list[int]:
    # get sorted shard dirs
    data = Path(run_dir) / Path(feature_type)
    shard_dirs = [dir for dir in sorted(data.glob("shard_*")) if dir.is_dir()]

    canonical_ids = []

    # read ids per shard
    for shard_dir in shard_dirs:
        # extract index
        shard_id = int(shard_dir.name.split("_")[1])
        _, ids_list = read_validate_shard(
            run_dir=run_dir, feature_type=feature_type, shard_id=shard_id
        )
        canonical_ids.extend(ids_list)

    # guard against duplicates
    if len(canonical_ids) != len(set(canonical_ids)):
        raise ValueError("Canonical id set contains duplicates")

    return canonical_ids


def align_distances(
    canonical_ids: list[int], backend_ids: list[int], backend_distances: np.ndarray
) -> np.ndarray:
    # ensure no duplicate ids in backend
    if len(backend_ids) != len(set(backend_ids)):
        raise ValueError("Backend id set contains duplicates")

    # ensure backend and canonical ids have equal coverage
    if set(backend_ids) != set(canonical_ids):
        raise ValueError("Backend and canonical id sets are not equal")

    # align ids & distances within dict
    distance_by_id = dict(zip(backend_ids, backend_distances, strict=True))

    # build aligned distance array
    aligned_distances = [distance_by_id[id] for id in canonical_ids]

    return np.array(aligned_distances, dtype=np.float32)
