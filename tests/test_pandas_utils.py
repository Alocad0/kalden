import numpy as np
import pandas as pd
import pytest

from kalden.core.datascience.pandas import (
    DataFrameUtils,
    DateTimeIndexUtils,
    SeriesUtils,
)


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


def test_fill_gaps_fills_only_blocks_within_limit() -> None:
    series = pd.Series([0.0, np.nan, np.nan, 3.0, np.nan, np.nan, np.nan, 7.0])

    result = SeriesUtils.fill_gaps(series, max_gap=2)

    assert result.iloc[:4].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert result.iloc[4:7].isna().all()
    assert result.iloc[7] == 7.0


def test_fill_gaps_supports_mean_and_constant_strategies() -> None:
    series = pd.Series([2.0, np.nan, 6.0])

    mean_result = SeriesUtils.fill_gaps(series, max_gap=1, method="mean")
    value_result = SeriesUtils.fill_gaps(
        series,
        max_gap=1,
        method="value",
        value=-1,
    )

    assert mean_result.tolist() == [2.0, 4.0, 6.0]
    assert value_result.tolist() == [2.0, -1.0, 6.0]


def test_detect_frequency_uses_most_common_timestep() -> None:
    index = pd.to_datetime(
        [
            "2024-01-01 00:00",
            "2024-01-01 01:00",
            "2024-01-01 02:00",
            "2024-01-01 04:00",
        ]
    )

    assert DateTimeIndexUtils.detect_frequency(index) == pd.Timedelta(hours=1)

    with pytest.raises(ValueError, match="stable frequency"):
        DateTimeIndexUtils.detect_frequency(index[:1])


def test_convert_numeric_like_columns_normalizes_mixed_locale_values() -> None:
    frame = pd.DataFrame(
        {
            "numeric": ["1,234.5", "1.234,5", "1 234,5", "(2,5)"],
            "mixed": ["1", "not numeric", "2", "3"],
            "excluded": ["1", "2", "3", "4"],
        }
    )

    result = DataFrameUtils.convert_numeric_like_columns(
        frame,
        exclude=["excluded"],
        normalize_numeric_strings=True,
    )

    assert result["numeric"].tolist() == [1234.5, 1234.5, 1234.5, -2.5]
    assert result["mixed"].tolist() == frame["mixed"].tolist()
    assert result["excluded"].tolist() == frame["excluded"].tolist()
    assert frame["numeric"].tolist() == [
        "1,234.5",
        "1.234,5",
        "1 234,5",
        "(2,5)",
    ]


def test_duplicate_index_counts_and_nan_summary_report_gaps() -> None:
    index = pd.to_datetime(
        ["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"]
    )
    frame = pd.DataFrame(
        {"value": [1.0, np.nan, np.nan, 4.0]},
        index=index,
    )

    duplicates = DataFrameUtils.duplicated_index_counts(frame)
    summary = DataFrameUtils.nan_summary(frame)

    assert duplicates.to_dict(orient="records") == [
        {"index_value": pd.Timestamp("2024-01-02"), "count": 2}
    ]
    assert summary.loc["value", "nan_count"] == 2
    assert summary.loc["value", "max_consecutive_nans"] == 2
    assert summary.loc["value", "max_gap_start"] == pd.Timestamp("2024-01-02")


def test_compute_volume_supports_step_and_trapezoidal_integration() -> None:
    frame = pd.DataFrame(
        {"Q": [5.0, 1.0, 3.0]},
        index=pd.to_datetime(
            ["2024-01-01 03:00", "2024-01-01 00:00", "2024-01-01 01:00"]
        ),
    )

    step_frame, step_total = DataFrameUtils.compute_volume(frame, method="step")
    trapezoid_frame, trapezoid_total = DataFrameUtils.compute_volume(
        frame,
        method="trapezoidal",
    )

    assert step_total["Q"] == pytest.approx(25_200)
    assert trapezoid_total["Q"] == pytest.approx(36_000)
    assert step_frame["cumulative_volume_m3", "Q"].iloc[-1] == pytest.approx(
        step_total["Q"]
    )
    assert trapezoid_frame["volume_m3", "Q"].iloc[0] == 0


def test_compute_volume_converts_litres_per_second() -> None:
    frame = pd.DataFrame(
        {"Q": [1000.0, 1000.0]},
        index=pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00"]),
    )

    _, total = DataFrameUtils.compute_volume(frame, unit="l/s")

    assert total["Q"] == pytest.approx(3600)


def test_reindex_to_hourly_adds_missing_timestamps() -> None:
    frame = pd.DataFrame(
        {"value": [1, 3]},
        index=pd.to_datetime(["2024-01-01 00:00", "2024-01-01 02:00"]),
    )

    result = DataFrameUtils.reindex_to_hourly(frame)

    assert result.index.tolist() == pd.date_range(
        "2024-01-01 00:00",
        periods=3,
        freq="h",
    ).tolist()
    assert pd.isna(result.loc["2024-01-01 01:00", "value"])
