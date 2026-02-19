from pathlib import Path


def run_extraction(
    feature_type: str,
    input_mode: str,
    run_dir: Path,
    shard_start: int | None,
    shard_stop: int | None,
    pilot_path: Path,
    db_path: Path,
    shard_size: int | None,
    policy: str,
) -> None:
    raise NotImplementedError
