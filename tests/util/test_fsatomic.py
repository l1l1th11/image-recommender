from pathlib import Path
from typing import BinaryIO

from image_recommender.util.fsatomic import write_tmp_then_rename

TEST_BYTES = bytes([74, 65, 73, 74])


def _write_bytes(f: BinaryIO) -> None:
    # test write function
    f.write(TEST_BYTES)


def test_happy_path(tmp_path: Path) -> None:
    # compose final path
    happy_path = tmp_path / "test_shard.bin"
    write_tmp_then_rename(final=happy_path, write_fn=_write_bytes)
    # convert to list
    happy_entries = list(tmp_path.iterdir())
    # check tmp file was replaced and final file was created
    assert len(happy_entries) == 1
    # check final file was created
    assert happy_entries[0].name == "test_shard.bin"
    # check final file contains expected bytes
    assert happy_path.read_bytes() == TEST_BYTES
