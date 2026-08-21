from pathlib import Path

import pandas as pd
import pytest

from kalden.core.mike import timeseries


def test_failed_in_place_rewrite_preserves_source(tmp_path, monkeypatch) -> None:
    source = tmp_path / "input.dfs0"
    source.write_bytes(b"original")
    reader = timeseries.Dfs0(source)

    class Dataset:
        n_items = 1
        items = [object()]

        def to_dataframe(self, **kwargs):
            return pd.DataFrame(
                {"value": [1.0]},
                index=pd.to_datetime(["2024-01-01"]),
            )

    class Rebuilt:
        def to_dfs(self, path, **kwargs):
            path.write_bytes(b"partial")
            raise OSError("simulated write failure")

    monkeypatch.setattr(reader, "validate_timestamps", lambda source: Dataset())
    monkeypatch.setattr(timeseries.mikeio, "from_pandas", lambda *args, **kwargs: Rebuilt())

    with pytest.raises(OSError, match="simulated write failure"):
        reader.rewrite()

    assert source.read_bytes() == b"original"
    assert list(tmp_path.glob(".*.dfs0")) == []


def test_source_resolution_requires_existing_dfs0_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be provided"):
        timeseries.Dfs0._resolve_source(None, None)

    text_file = tmp_path / "input.txt"
    text_file.touch()
    with pytest.raises(ValueError, match="Expected a .dfs0 file"):
        timeseries.Dfs0._resolve_source(text_file, None)

    with pytest.raises(FileNotFoundError, match="not found"):
        timeseries.Dfs0._resolve_source(tmp_path / "missing.dfs0", None)


def test_destination_resolution_creates_parent_and_guards_overwrite(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.dfs0"
    source.touch()
    destination = tmp_path / "nested" / "output.dfs0"

    result = timeseries.Dfs0._resolve_destination(
        source,
        destination,
        overwrite=False,
    )
    assert result == destination
    assert destination.parent.is_dir()

    destination.touch()
    with pytest.raises(FileExistsError, match="already exists"):
        timeseries.Dfs0._resolve_destination(
            source,
            destination,
            overwrite=False,
        )
    assert (
        timeseries.Dfs0._resolve_destination(source, destination, overwrite=True)
        == destination
    )

    with pytest.raises(ValueError, match="Expected a .dfs0 file"):
        timeseries.Dfs0._resolve_destination(
            source,
            tmp_path / "output.csv",
            overwrite=True,
        )


def test_iter_files_is_sorted_and_respects_recursive_flag(tmp_path: Path) -> None:
    first = tmp_path / "a.dfs0"
    second = tmp_path / "b.dfs0"
    nested = tmp_path / "nested" / "c.dfs0"
    (tmp_path / "nested").mkdir()
    for path in (second, nested, first):
        path.touch()
    (tmp_path / "ignored.txt").touch()

    assert timeseries.Dfs0.iter_files(tmp_path, recursive=False) == [first, second]
    assert timeseries.Dfs0.iter_files(tmp_path) == [first, second, nested]
    assert timeseries.Dfs0.iter_files(first) == [first]


def test_validate_item_count_preserves_defaults_and_checks_replacements() -> None:
    class Dataset:
        n_items = 2
        items = ["first", "second"]

    assert timeseries.Dfs0._validate_item_count(Dataset(), None) == [
        "first",
        "second",
    ]
    assert timeseries.Dfs0._validate_item_count(Dataset(), [1, 2]) == [1, 2]

    with pytest.raises(ValueError, match="must match"):
        timeseries.Dfs0._validate_item_count(Dataset(), [1])


@pytest.mark.parametrize(
    ("timestamps", "n_timesteps", "message"),
    [
        ([], 0, "no time steps"),
        (
            ["2024-01-02", "2024-01-01"],
            2,
            "not sorted in ascending order",
        ),
        (
            ["2024-01-01", "2024-01-01"],
            2,
            "Duplicate timestamps",
        ),
    ],
)
def test_validate_timestamps_rejects_invalid_time_axes(
    tmp_path: Path,
    monkeypatch,
    timestamps,
    n_timesteps: int,
    message: str,
) -> None:
    source = tmp_path / "input.dfs0"
    source.touch()
    reader = timeseries.Dfs0(source)

    class Dataset:
        time = pd.DatetimeIndex(timestamps)

    Dataset.n_timesteps = n_timesteps
    monkeypatch.setattr(reader, "read", lambda path: Dataset())

    with pytest.raises(ValueError, match=message):
        reader.validate_timestamps()


def test_scan_duplicate_timestamps_skips_excluded_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    included = tmp_path / "included.dfs0"
    clean = tmp_path / "clean.dfs0"
    excluded = tmp_path / "archive" / "excluded.dfs0"
    excluded.parent.mkdir()
    for path in (included, clean, excluded):
        path.touch()

    duplicate = pd.Timestamp("2024-01-01")

    def fake_duplicates(self):
        if self.path in {included, excluded}:
            return pd.DatetimeIndex([duplicate, duplicate])
        return pd.DatetimeIndex([])

    monkeypatch.setattr(timeseries.Dfs0, "duplicate_timestamps", fake_duplicates)

    result = timeseries.Dfs0.scan_duplicate_timestamps(
        tmp_path,
        exclude_substrings=["archive"],
    )

    assert result == {included: [duplicate, duplicate]}
