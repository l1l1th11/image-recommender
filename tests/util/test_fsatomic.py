from pathlib import Path
from typing import BinaryIO

import pytest

from image_recommender.util.fsatomic import write_tmp_then_rename

TEST_BYTES = bytes([74, 65, 73, 74])


def _write_bytes(f: BinaryIO) -> None:
    """
    Writes deterministic payload for test.
    """
    f.write(TEST_BYTES)


def _write_error(f: BinaryIO) -> None:
    """
    Writes partial data, then raises to simulate write failure.
    """
    f.write(bytes([11, 11]))
    raise RuntimeError("Intentional failure.")


def test_happy_path(tmp_path: Path) -> None:
    # compose final path
    happy_path = tmp_path / "happy_shard.bin"
    # write bytes
    write_tmp_then_rename(final=happy_path, write_fn=_write_bytes)
    # convert to list
    happy_entries = list(tmp_path.iterdir())
    # check only one file exists (no tmp leftover)
    assert len(happy_entries) == 1
    # check final file was created
    assert happy_entries[0].name == "happy_shard.bin"
    # check final file contains expected bytes
    assert happy_path.read_bytes() == TEST_BYTES


def test_no_final_on_failing_w(tmp_path: Path) -> None:
    # compose final path
    missing_path = tmp_path / "missing_shard.bin"
    # fail write
    with pytest.raises(RuntimeError):
        write_tmp_then_rename(final=missing_path, write_fn=_write_error)
    # check final was not created
    assert missing_path.exists() is False


def test_no_updated_final_on_failing_w(tmp_path: Path) -> None:
    # compose final path
    old_path = tmp_path / "old_shard.bin"
    # write bytes
    write_tmp_then_rename(final=old_path, write_fn=_write_bytes)
    # fail second write
    with pytest.raises(RuntimeError):
        write_tmp_then_rename(final=old_path, write_fn=_write_error)
    # check final was not updated on failed write
    assert old_path.read_bytes() == TEST_BYTES
