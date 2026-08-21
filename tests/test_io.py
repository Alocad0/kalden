from pathlib import Path

import pytest

from kalden.core import io


def test_directory_helpers_create_and_empty_nested_content(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "directory"
    io.ensure_dir_exists(target)
    assert target.is_dir()
    assert io.is_dir_empty(target)

    io.ensure_file_dir_exists(target / "child" / "data.txt")
    (target / "file.txt").write_text("data", encoding="utf-8")
    (target / "child" / "data.txt").write_text("data", encoding="utf-8")
    assert not io.is_dir_empty(target)

    io.empty_dir(target)
    assert target.is_dir()
    assert io.is_dir_empty(target)


def test_empty_dir_validates_the_target(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    io.empty_dir(missing, missing_ok=True)

    with pytest.raises(FileNotFoundError, match="does not exist"):
        io.empty_dir(missing)

    file_path = tmp_path / "file.txt"
    file_path.write_text("data", encoding="utf-8")
    with pytest.raises(NotADirectoryError, match="not a directory"):
        io.empty_dir(file_path)


def test_temporary_directory_lifecycle_uses_requested_parent(tmp_path: Path) -> None:
    temp_dir = io.create_temp_dir(prefix="kalden_", base_dir=tmp_path)

    assert temp_dir.parent == tmp_path
    assert temp_dir.name.startswith("kalden_")
    assert temp_dir.is_dir()

    (temp_dir / "nested").mkdir()
    (temp_dir / "nested" / "file.txt").write_text("data", encoding="utf-8")
    io.delete_temp_dir(temp_dir)
    assert not temp_dir.exists()


def test_file_helpers_distinguish_missing_empty_and_populated_files(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.txt"
    empty = tmp_path / "empty.txt"
    populated = tmp_path / "populated.txt"
    empty.touch()
    populated.write_text("content", encoding="utf-8")

    assert not io.file_exists(missing)
    assert io.file_exists(empty)
    assert not io.file_has_content(empty)
    assert io.file_has_content(populated)
    assert not io.file_exists(tmp_path)


def test_detect_file_encoding_reads_bom_encoded_text(tmp_path: Path) -> None:
    path = tmp_path / "utf16.txt"
    path.write_text("Kalden encoding test", encoding="utf-16")

    encoding = io.detect_file_encoding(path)

    assert encoding is not None
    assert encoding.lower().replace("_", "-") == "utf-16"
