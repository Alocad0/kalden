"""
Utility functions for generic disk writing / reading operations.

This module provides helper methods used across projects.

Author: ALDE / DEAO
Created: 2026-01-15
"""

import os
import chardet
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]


def hello(name: str) -> str:
    """For testing purposes."""
    return f"Hello, {name}!"

# -------------------------  DIRECTORY  ------------------------------------

def ensure_dir_exists(dir_path: PathLike) -> None:
    """
    Ensure that the directory path exists.
    If the directory (or any of its parents) does not exist, it is created.

    Parameters
    ----------
    dir_path : str | Path
        Path to a directory that should be ensured to exist.
    """
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        dir_path.mkdir(parents=True, exist_ok=True)


def ensure_file_dir_exists(file_path: PathLike) -> None:
    """
    Ensure that the parent directory of the given file path exists.
    If the directory (or any of its parents) does not exist, it is created.

    Parameters
    ----------
    file_path : str | Path
        Path to a file whose parent directory should be ensured to exist.
    """
    dir_path = Path(file_path).parent
    ensure_dir_exists(dir_path)


def is_dir_empty(dir_path: PathLike) -> bool:
    """Return True if the directory is empty, False otherwise."""
    return len(os.listdir(dir_path)) == 0


# def empty_dir(folder_path):
#     folder = Path(folder_path)
#
#     if not folder.exists():
#         raise FileNotFoundError(f"Folder does not exist: {folder}")
#
#     if not folder.is_dir():
#         raise NotADirectoryError(f"Not a folder: {folder}")
#
#     for item in folder.iterdir():
#         if item.is_dir():
#             shutil.rmtree(item)
#         else:
#             item.unlink()


def empty_dir(dir_path: PathLike, missing_ok: bool = False) -> None:
    """
    Remove all contents of a directory without deleting the directory itself.

    Parameters
    ----------
    dir_path : str | Path
        Path to the directory to empty.
    missing_ok : bool, default=False
        If True, do nothing when the directory does not exist.
        If False, raise FileNotFoundError.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist and `missing_ok` is False.
    NotADirectoryError
        If `dir_path` exists but is not a directory.
    """
    path = Path(dir_path)

    if not path.exists():
        if missing_ok:
            return
        raise FileNotFoundError(f"Directory does not exist: {path}")

    if not path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {path}")

    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

def create_temp_dir(prefix: str = "tmp_", base_dir: Optional[PathLike] = None) -> Path:
    """
    Create a temporary directory and return its path.

    Parameters
    ----------
    prefix : str, optional
        Prefix for the temporary directory name.
    base_dir : str or Path, optional
        Parent directory where the temp directory should be created.
        If None, the system temp location is used.

    Returns
    -------
    Path
        Path to the created temporary directory.
    """
    temp_dir = tempfile.mkdtemp(prefix=prefix, dir=str(base_dir) if base_dir else None)
    return Path(temp_dir)


def delete_temp_dir(temp_dir: PathLike, ignore_errors: bool = False) -> None:
    """
    Delete a temporary directory and all its contents.

    Parameters
    ----------
    temp_dir : str | Path
        Path to the temporary directory.
    ignore_errors : bool, optional
        If True, suppress deletion errors.
    """
    shutil.rmtree(Path(temp_dir), ignore_errors=ignore_errors)


# -------------------------  FILE  ------------------------------------

def file_exists(file_path: PathLike) -> bool:
    """
    Check whether a file exists at the given path.

    Parameters
    ----------
    file_path : str | Path
        Path to the file to check.

    Returns
    -------
    bool
        True if the file exists and is a regular file, otherwise False.
    """
    return Path(file_path).is_file()

def file_has_content(file_path: str | Path) -> bool:
    """
    Detects if file is exists / is empty
    Move to io.py
    """
    path = Path(file_path)
    return path.exists() and path.is_file() and path.stat().st_size > 0

def detect_file_encoding(file_path: PathLike) -> Optional[str]:
    """
    Detect the text encoding of a file.

    Parameters
    ----------
    file_path : str | Path
        Path to the file to inspect.

    Returns
    -------
    str | None
        The detected encoding, or None if it cannot be determined.
    """
    with open(file_path, "rb") as f:
        raw_data = f.read(1024 * 100)
        result = chardet.detect(raw_data)
    return result.get("encoding")

