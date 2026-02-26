from pathlib import Path

import pytest

from image_recommender.cli.main import main
from image_recommender.config import DB_PATH, PILOT_IDS_CSV
from image_recommender.db.connector import get_any_image_path


@pytest.mark.integration
def test_pilot_hsv_extraction(tmp_path: Path):
    # ensure pilot and db exist
    if not PILOT_IDS_CSV.exists():
        pytest.skip(f"Pilot csv file at {str(PILOT_IDS_CSV)} is missing")
    if not DB_PATH.exists():
        pytest.skip(f"Database at {str(DB_PATH)} is missing")

    # ensure db has entries and images can be accessed
    sample_path = get_any_image_path(db_path=DB_PATH)
    if sample_path is None:
        pytest.skip("No images present in database")
    if not Path(sample_path).exists():
        pytest.skip("Sample image can't be opened, external hard drive possibly not connected")

    # call cli directly for hsv extraction on pilot
    exit_code = main(
        [
            "extract",
            "--feature-type",
            "hsv",
            "--input-mode",
            "pilot",
            "--run-dir",
            str(tmp_path),
            "--policy",
            "skip_and_log",
        ]
    )
    assert exit_code == 0

    hsv_dir = tmp_path / "hsv"
    assert hsv_dir.exists()

    mtimes_1 = {}

    # collect shard dirs
    shards_run_1 = sorted(
        [p for p in hsv_dir.iterdir() if p.is_dir() and p.name.startswith("shard_")]
    )

    # check at least 1 shard created
    assert len(shards_run_1) > 0

    # ensure shard dirs contain expected files
    for shard_dir in shards_run_1:
        assert shard_dir.name.startswith("shard_")
        features_path = shard_dir / "features.npy"
        assert features_path.exists()
        ids_path = shard_dir / "ids.npy"
        assert ids_path.exists()
        meta_path = shard_dir / "meta.json"
        assert meta_path.exists()
        marker_path = shard_dir / "_SUCCESS"
        assert marker_path.exists()

        # collect modification time
        mtime = features_path.stat().st_mtime
        mtimes_1[shard_dir.name] = mtime

    # rerun hsv extraction
    exit_code_2 = main(
        [
            "extract",
            "--feature-type",
            "hsv",
            "--input-mode",
            "pilot",
            "--run-dir",
            str(tmp_path),
            "--policy",
            "skip_and_log",
        ]
    )
    assert exit_code_2 == 0

    mtimes_2 = {}

    shards_run_2 = sorted(
        [p for p in hsv_dir.iterdir() if p.is_dir() and p.name.startswith("shard_")]
    )

    # collect modification times
    for shard_dir in shards_run_2:
        features_path = shard_dir / "features.npy"
        mtime = features_path.stat().st_mtime
        mtimes_2[shard_dir.name] = mtime

    # ensure modification times are unchanged and no extra shards were created
    assert mtimes_1 == mtimes_2
