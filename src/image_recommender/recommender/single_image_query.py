from pathlib import Path

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
