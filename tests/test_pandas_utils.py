import pandas as pd
import pytest

from kalden.core.datascience.pandas import DataFrameUtils


def test_smart_resample_supports_current_pandas_offsets() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="h")
    frame = pd.DataFrame({"value": [0.0, 1.0, 2.0, 3.0]}, index=index)

    upsampled = DataFrameUtils.smart_resample(frame, "30min")
    downsampled = DataFrameUtils.smart_resample(frame, "2h")

    assert len(upsampled) == 7
    assert downsampled["value"].tolist() == [0.5, 2.5]


def test_smart_resample_accepts_calendar_frequency() -> None:
    index = pd.date_range("2024-01-01", periods=40, freq="D")
    frame = pd.DataFrame({"value": range(40)}, index=index)

    result = DataFrameUtils.smart_resample(frame, "ME")

    assert len(result) == 2


def test_duplicate_year_drops_leap_day_by_default() -> None:
    index = pd.to_datetime(["2024-02-28", "2024-02-29", "2024-03-01"])
    frame = pd.DataFrame({"value": [1, 2, 3]}, index=index)

    result = DataFrameUtils.duplicate_year(frame, 2023, 2024)

    has_2023_leap_day = (
        (result.index.year == 2023)
        & (result.index.month == 2)
        & (result.index.day == 29)
    ).any()
    assert not has_2023_leap_day
    assert pd.Timestamp("2024-02-29") in result.index
    assert len(result) == 5


def test_duplicate_year_supports_explicit_leap_day_policy() -> None:
    frame = pd.DataFrame(
        {"value": [2]},
        index=pd.to_datetime(["2024-02-29"]),
    )

    result = DataFrameUtils.duplicate_year(
        frame,
        2023,
        2023,
        leap_day="feb28",
    )

    assert result.index.tolist() == [pd.Timestamp("2023-02-28")]

    with pytest.raises(ValueError, match="February 29"):
        DataFrameUtils.duplicate_year(frame, 2023, 2023, leap_day="raise")
