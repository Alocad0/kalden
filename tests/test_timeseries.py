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
