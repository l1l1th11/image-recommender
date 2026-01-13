import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO


def write_tmp_then_rename(final: Path | str, write_fn: Callable[[BinaryIO], None]) -> None:
    """
    Write temporary file to same dir as final, using a callable, then atomic replace.
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

    # atomic replace
    os.replace(tmp_path, final)
