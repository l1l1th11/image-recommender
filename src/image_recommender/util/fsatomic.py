import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO


def write_tmp_then_rename(final: Path | str, write_fn: Callable[[BinaryIO], None]) -> None:
    """
    Writes to named temporary file in same directory as final using a callable,
    flush & fsync, then atomically replace final with tmp file contents.
    Best-effort: fsync parent directory.

    Inputs:
    - final: path to final file
    - write_fn: callable to write to file

    Output: None (writes to disk)
    """

    # convert to path object
    final = Path(final)
    # check parent dir exists
    parent_dir = final.parent
    if not parent_dir.exists():
        raise FileNotFoundError("Parent directory does not exist.")
    # create named temporary file in final files parent dir
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=final.name + ".tmp-", dir=parent_dir, delete=False
    ) as tmp_file:
        # capture temporary files path and convert
        tmp_path = Path(tmp_file.name)

        # pass open file handler to write function
        write_fn(tmp_file)

        # flush bytes in pythons buffer to os
        tmp_file.flush()

        # fsync: request os to sync file contents to disk
        os.fsync(tmp_file.fileno())

    # atomic replace
    os.replace(tmp_path, final)

    # best effort: fsync parent dir
    try:
        # open parent dir file descriptor
        dir_fd = os.open(str(parent_dir), os.O_RDONLY)

        try:
            # fsync: request os to sync dir contents to disk
            os.fsync(dir_fd)

        finally:
            # close file descriptor
            os.close(dir_fd)

    except (OSError, TypeError, AttributeError):
        # fail silently
        pass
