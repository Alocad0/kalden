import pandas as pd
import pytest

from kalden.core.mike.res1d import DEFAULT_OBJECT_TYPES, Res1D, SeriesRef


def test_cache_is_invalidated_when_source_changes(tmp_path) -> None:
    source = tmp_path / "result.res1d"
    source.write_bytes(b"first")
    reader = Res1D(source, cache_dir=tmp_path / "cache")
    stem = reader.cache_path("node", "N1", "WaterLevel")
    frame = pd.DataFrame(
        {"value": [1.0]},
        index=pd.to_datetime(["2024-01-01"]),
    )

    reader._write_dataframe_cache(frame, stem)
    assert reader._read_dataframe_cache(stem) is not None

    source.write_bytes(b"second version")

    assert reader._read_dataframe_cache(stem) is None


def test_clear_cache_refuses_unowned_custom_directory(tmp_path) -> None:
    source = tmp_path / "result.res1d"
    source.write_bytes(b"result")
    cache_dir = tmp_path / "shared-data"
    cache_dir.mkdir()
    important = cache_dir / "important.txt"
    important.write_text("keep", encoding="utf-8")
    reader = Res1D(source, cache_dir=cache_dir)

    with pytest.raises(ValueError, match="Refusing to remove"):
        reader.clear_cache()
    assert important.is_file()


def test_every_declared_object_type_has_a_spatial_cache_stem(tmp_path) -> None:
    reader = Res1D(tmp_path / "result.res1d")

    stems = {kind: reader._spatial_cache_stem(kind) for kind in DEFAULT_OBJECT_TYPES}

    assert set(stems) == set(DEFAULT_OBJECT_TYPES)
    assert len(set(stems.values())) == len(DEFAULT_OBJECT_TYPES)


def test_auto_load_mode_uses_file_size_threshold(tmp_path) -> None:
    source = tmp_path / "result.res1d"
    source.write_bytes(b"12345")

    assert Res1D(source, full_load_max_bytes=5).effective_load_mode == "full"
    assert Res1D(source, full_load_max_bytes=4).effective_load_mode == "filtered"


def test_multi_chainage_total_requires_explicit_selection(tmp_path, monkeypatch) -> None:
    reader = Res1D(tmp_path / "result.res1d", cache=False)
    frame = pd.DataFrame(
        {"Q:reach:0": [1.0], "Q:reach:10": [2.0]},
        index=pd.to_datetime(["2024-01-01"]),
    )
    monkeypatch.setattr(reader, "read_series", lambda *args, **kwargs: frame)

    with pytest.raises(ValueError, match="multi-chainage"):
        reader.combine_series(
            refs=[SeriesRef("reach", "R1", "Discharge")],
            chainage="all",
        )
