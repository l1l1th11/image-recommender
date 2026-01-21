from pathlib import Path

VERSION = 1

REQUIRED_META_KEYS = {
    "feature_type",
    "feature_dim",
    "feature_dtype",
    "shard_size",
    "created_at",
    "version",
}


def shard_paths(run_dir: Path | str, feature_type: str, shard_id: int) -> tuple[Path, Path, Path]:
    """
    Return artifact paths for one shard in a run directory.

    Example:
        >>> shard_paths("runs/run_1", "hsv", 1)
        (Path("runs/run_1/hsv/shard_0001/features.npy"),
         Path("runs/run_1/hsv/shard_0001/ids.npy"),
         Path("runs/run_1/hsv/shard_0001/meta.json"))

    Notes:
        See required metadata keys in REQUIRED_META_KEYS.
    """
    # convert to path
    run_dir = Path(run_dir)
    # build shard directory
    shard_dir = run_dir / feature_type / f"shard_{shard_id:04d}"  # 4 digit zero padding
    # construct artifact paths
    features_path = shard_dir / "features.npy"
    ids_path = shard_dir / "ids.npy"
    meta_path = shard_dir / "meta.json"

    return features_path, ids_path, meta_path
