"""Generic pandas helpers used across projects.

The module is organized by pandas object type:

* ``SeriesUtils`` for ``pandas.Series`` helpers
* ``DateTimeIndexUtils`` for ``pandas.DatetimeIndex`` helpers
* ``DataFrameUtils`` for ``pandas.DataFrame`` helpers

Thin function wrappers are kept at module level for convenience and
compatibility with earlier utility-style usage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

__all__ = [
    "SeriesUtils",
    "DateTimeIndexUtils",
    "DataFrameUtils",
    "df_col_to_numeric",
    "series_fill_gaps",
    "series_recycle_gaps",
    "df_duplicated_index_counts",
    "df_nan_summary",
    "df_check_duplicates",
    "df_time_index_summary",
    "df_resample",
    "df_smart_resample",
    "df_detect_frequency",
    "df_reindex_to_hourly",
    "df_duplicate_year",
    "df_compute_volume",
    "df_split_column_by_distribution",
    "df_split_column_equally",
    "df_plot",
    "df_columns_to_numeric",
]


class SeriesUtils:
    """Helpers operating on pandas Series objects."""

    @staticmethod
    def to_numeric(col: pd.Series) -> pd.Series:
        """Convert a Series to numeric where possible, keeping original values otherwise."""

        def try_convert(value):
            try:
                return pd.to_numeric(value)
            except (ValueError, TypeError):
                return value

        return col.apply(try_convert)

    @staticmethod
    def fill_gaps(
        col: pd.Series,
        max_gap: int,
        method: str = "interpolate",
        value=None,
    ) -> pd.Series:
        """
        Fill short NaN gaps in a Series.

        Parameters
        ----------
        col : pd.Series
            Series to process.
        max_gap : int
            Maximum number of consecutive NaNs to fill.
        method : {"interpolate", "mean", "value"}, default "interpolate"
            Fill strategy.
        value : object, optional
            Constant fill value used when ``method="value"``.
        """
        if method not in {"interpolate", "mean", "value"}:
            raise ValueError("method must be either 'interpolate', 'mean' or 'value'")

        s = col.copy()
        is_nan = s.isna()
        grp = (~is_nan).cumsum()
        nan_block_sizes = is_nan.groupby(grp).sum()

        for block_id, block_size in nan_block_sizes.items():
            if block_size == 0 or block_size > max_gap:
                continue

            mask = (grp == block_id) & is_nan
            block_positions = np.where(mask)[0]
            start_pos = block_positions[0]
            end_pos = block_positions[-1]

            left_pos = start_pos - 1 if start_pos > 0 else None
            right_pos = end_pos + 1 if end_pos < len(s) - 1 else None

            left_val = s.iloc[left_pos] if left_pos is not None else np.nan
            right_val = s.iloc[right_pos] if right_pos is not None else np.nan

            if method == "interpolate":
                s.loc[mask] = np.nan
                s_interp = s.interpolate(method="linear", limit_direction="both")
                s.loc[mask] = s_interp.loc[mask]

            elif method == "mean":
                if not np.isnan(left_val) and not np.isnan(right_val):
                    s.loc[mask] = (left_val + right_val) / 2.0
                elif not np.isnan(left_val):
                    s.loc[mask] = left_val
                elif not np.isnan(right_val):
                    s.loc[mask] = right_val

            elif method == "value":
                if pd.isna(value):
                    raise ValueError("Value must be different from None or np.nan")
                s.loc[mask] = value

        return s

    @staticmethod
    def recycle_gaps(
        s: pd.Series,
        period: str = "daily",
        method: str = "resample",
    ) -> pd.Series:
        """
        Fill missing values using averages for the same calendar period across years.

        Parameters
        ----------
        s : pd.Series
            Series with a datetime index.
        period : {"daily", "hourly", "monthly"}, default "daily"
            Calendar grouping to use.
        method : {"resample", "groupby"}, default "resample"
            Strategy used to compute the reference seasonal averages.
        """
        s = s.copy()
        s.index = pd.to_datetime(s.index)

        if period not in ("daily", "hourly", "monthly"):
            raise ValueError("period must be one of ['daily', 'hourly', 'monthly']")

        if method not in ("resample", "groupby"):
            raise ValueError("method must be one of ['resample', 'groupby']")

        if method == "resample":
            if period == "daily":
                daily = s.resample("D").mean()
                group_means = daily.groupby([daily.index.month, daily.index.day]).mean()
                group_means.index.names = ["month", "day"]
            elif period == "hourly":
                hourly = s.resample("h").mean()
                group_means = hourly.groupby(
                    [hourly.index.month, hourly.index.day, hourly.index.hour]
                ).mean()
                group_means.index.names = ["month", "day", "hour"]
            else:
                monthly = s.resample("ME").mean()
                group_means = monthly.groupby(monthly.index.month).mean()
                group_means.index.names = ["month"]

            df = pd.DataFrame({"value": s})
            df["month"] = df.index.month
            df["day"] = df.index.day
            if period == "hourly":
                df["hour"] = df.index.hour

            def fill_na(row):
                if pd.isna(row["value"]):
                    try:
                        if period == "daily":
                            return group_means.loc[(row["month"], row["day"])]
                        if period == "hourly":
                            return group_means.loc[
                                (row["month"], row["day"], row["hour"])
                            ]
                        return group_means.loc[row["month"]]
                    except KeyError:
                        return np.nan
                return row["value"]

            df["value"] = df.apply(fill_na, axis=1)
            return df["value"]

        df = pd.DataFrame({"value": s})
        df["month"] = df.index.month
        if period in ("daily", "hourly"):
            df["day"] = df.index.day
        if period == "hourly":
            df["hour"] = df.index.hour

        if period == "daily":
            group_means = df.groupby(["month", "day"])["value"].mean()
            df["value"] = df.apply(
                lambda row: group_means.loc[(row["month"], row["day"])]
                if pd.isna(row["value"])
                else row["value"],
                axis=1,
            )
        elif period == "hourly":
            group_means = df.groupby(["month", "day", "hour"])["value"].mean()
            df["value"] = df.apply(
                lambda row: group_means.loc[(row["month"], row["day"], row["hour"])]
                if pd.isna(row["value"])
                else row["value"],
                axis=1,
            )
        else:
            group_means = df.groupby("month")["value"].mean()
            df["value"] = df.apply(
                lambda row: group_means.loc[row["month"]]
                if pd.isna(row["value"])
                else row["value"],
                axis=1,
            )

        return df["value"]


class DateTimeIndexUtils:
    """Helpers operating on pandas DatetimeIndex objects."""

    @staticmethod
    def detect_frequency(index: pd.DatetimeIndex):
        """
        Detect a stable frequency based on the mode of index time differences.
        """
        diffs = index.to_series().diff().dropna()
        freq = diffs.mode()
        if len(freq) == 0:
            raise ValueError("Cannot detect a stable frequency from index.")
        return freq.iloc[0]


class DataFrameUtils:
    """Helpers operating on pandas DataFrame objects."""

    @staticmethod
    def duplicated_index_counts(df: pd.DataFrame) -> pd.DataFrame:
        """Return duplicate index values and their occurrence counts."""
        index_counts = df.index.value_counts()
        duplicates = index_counts[index_counts > 1]

        if duplicates.empty:
            return pd.DataFrame(columns=["index_value", "count"])

        result_df = duplicates.reset_index()
        result_df.columns = ["index_value", "count"]
        return result_df

    @staticmethod
    def nan_summary(df: pd.DataFrame) -> pd.DataFrame:
        """Summarize NaN counts and largest NaN gaps for each column."""

        def get_max_gap_duration(gap_start, gap_end):
            if (gap_start is not None) and (gap_end is not None):
                td = gap_end - gap_start
                return "{:.0f} days {:.0f} hours".format(td.days, td.seconds // 3600)
            return None

        summary = {}

        for col in df.columns:
            s = df[col]
            is_nan = s.isna()
            nan_count = is_nan.sum()
            group_id = (is_nan != is_nan.shift()).cumsum()
            grouped = is_nan.groupby(group_id)

            max_gap = 0
            max_start = None
            max_end = None

            for _, group in grouped:
                if group.all():
                    gap_size = len(group)
                    if gap_size > max_gap:
                        max_gap = gap_size
                        max_start = group.index[0]
                        max_end = group.index[-1]

            summary[col] = {
                "nan_count": int(nan_count),
                "max_consecutive_nans": int(max_gap),
                "max_gap_start": max_start,
                "max_gap_end": max_end,
            }

            if isinstance(df.index, pd.DatetimeIndex):
                summary[col]["max_gap_duration_days"] = get_max_gap_duration(
                    max_start,
                    max_end,
                )

        return pd.DataFrame(summary).T

    @staticmethod
    def check_duplicates(
        df: pd.DataFrame,
        name: str | None = None,
        show: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """Check duplicate index entries and NaN patterns for a DataFrame."""
        if name is None:
            name = "df"

        print("Checking for duplicates...")

        duplicates = DataFrameUtils.duplicated_index_counts(df)
        nans = DataFrameUtils.nan_summary(df)

        if duplicates.shape[0] > 0:
            print(f"\tDuplicates found: {duplicates.shape[0]}")
            if show:
                print(duplicates)

        if nans.nan_count.unique().shape[0] > 1:
            print("\tNaN summary:")
            print(nans)

        return {"duplicates": duplicates, "nans": nans}

    @staticmethod
    def time_index_summary(df: pd.DataFrame, plot: bool = False) -> pd.DataFrame:
        """Summarize the time-step characteristics of a DataFrame with a DateTimeIndex."""
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a pandas DateTimeIndex")

        deltas = df.index.to_series().diff().dropna()
        delta_seconds = deltas.dt.total_seconds()

        avg_step = pd.to_timedelta(delta_seconds.mean(), unit="s")
        min_step = pd.to_timedelta(delta_seconds.min(), unit="s")
        max_step = pd.to_timedelta(delta_seconds.max(), unit="s")
        avg_freq = 1 / delta_seconds.mean() if len(delta_seconds) > 0 else float("nan")

        summary = pd.DataFrame(
            {
                "n_points": [len(df)],
                "start": [df.index.min()],
                "end": [df.index.max()],
                "duration": [df.index.max() - df.index.min()],
                "average_timestep": [avg_step],
                "min_timestep": [min_step],
                "max_timestep": [max_step],
                "average_frequency (Hz)": [avg_freq],
            }
        )

        if plot:
            fig = px.line(
                x=deltas.index,
                y=delta_seconds,
                labels={"x": "Timestamp", "y": "dt (seconds)"},
                title="Time Step Differences (Index.diff())",
            )
            fig.update_traces(mode="lines+markers")
            fig.show()

        return summary.T

    @staticmethod
    def resample(
        df: pd.DataFrame,
        freq: str,
        method: str = "linear",
        limit_direction: str = "both",
        plot: bool = False,
        plot_column: str = "",
    ) -> pd.DataFrame:
        """Resample a time-indexed DataFrame to a target frequency with interpolation."""
        df = df.copy()

        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a pandas DateTimeIndex")

        df_resampled = df.resample(freq).mean()

        try:
            df_resampled = df_resampled.interpolate(
                method=method,
                limit_direction=limit_direction,
            )
        except Exception as exc:
            raise ValueError(f"Interpolation failed: {exc}") from exc

        if plot:
            if plot_column == "":
                raise ValueError("'plot_column' must be specified")
            if plot_column not in df.columns:
                raise ValueError(f"'{plot_column}' does not exist in the dataframe")

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[plot_column],
                    mode="lines+markers",
                    name="Raw",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df_resampled.index,
                    y=df_resampled[plot_column],
                    name="Interpolated",
                )
            )
            fig.show()

        return df_resampled

    @staticmethod
    def smart_resample(
        df: pd.DataFrame,
        target_freq: str,
        interpolate_method: str = "linear",
        limit_direction: str = "both",
    ) -> pd.DataFrame:
        """
        Choose a resampling strategy automatically based on source vs target frequency.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a pandas DateTimeIndex")

        df = df.copy()

        try:
            src_td = DateTimeIndexUtils.detect_frequency(df.index)
        except Exception as exc:
            raise ValueError(f"Failed to detect frequency: {exc}") from exc

        tgt_td = pd.tseries.frequencies.to_offset(target_freq).delta

        if src_td == tgt_td:
            return df.copy()

        if tgt_td > src_td:
            return df.resample(target_freq).mean()

        df_up = df.resample(target_freq).asfreq()
        df_up = df_up.interpolate(
            method=interpolate_method,
            limit_direction=limit_direction,
        )
        return df_up

    @staticmethod
    def reindex_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
        """Reindex a DataFrame to a complete hourly DateTimeIndex."""
        df = df.copy()
        df.index = pd.to_datetime(df.index)

        full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="h")
        return df.reindex(full_idx)

    @staticmethod
    def duplicate_year(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
        """Duplicate a DataFrame with a DateTimeIndex across a year range."""
        dfs = []
        for year in range(start_year, end_year + 1):
            df_copy = df.copy()
            df_copy.index = df_copy.index.map(lambda d: d.replace(year=year))
            dfs.append(df_copy)

        return pd.concat(dfs)

    @staticmethod
    def compute_volume(
        df: pd.DataFrame,
        discharge_column: str = "Q",
        volume_column: str | None = None,
        cumsum_column: str | None = None,
    ) -> pd.DataFrame:
        """Add per-timestep and cumulative volume columns based on a discharge column."""
        if volume_column is None:
            volume_column = discharge_column + "_volume"

        if cumsum_column is None:
            cumsum_column = discharge_column + "_volume_cumsum"

        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Index must be a DatetimeIndex")

        df = df.copy()
        dt = df.index.to_series().diff().dt.total_seconds()
        df[volume_column] = df[discharge_column] * dt
        df.iloc[0, df.columns.get_loc(volume_column)] = float("nan")
        df[cumsum_column] = df[volume_column].cumsum()
        return df

    @staticmethod
    def split_column_by_distribution(
        df: pd.DataFrame,
        column: str,
        percentages,
        new_names,
        drop_original: bool = False,
        decimals: int = 2,
    ) -> pd.DataFrame:
        """Split a numeric column into multiple new columns using a distribution."""
        df = df.copy()

        if len(percentages) != len(new_names):
            raise ValueError("`percentages` and `new_names` must have the same length.")

        total = sum(percentages)
        if total == 0:
            raise ValueError("Sum of percentages cannot be zero.")

        percentages = [p / total for p in percentages]

        for perc, name in zip(percentages, new_names):
            df[name] = (df[column] * perc).round(decimals)

        if drop_original:
            df = df.drop(columns=[column])

        return df

    @staticmethod
    def split_column_equally(
        df: pd.DataFrame,
        column: str,
        new_names,
        drop_original: bool = False,
        decimals: int = 4,
    ) -> pd.DataFrame:
        """Duplicate a numeric column into multiple identical rounded columns."""
        for name in new_names:
            df[name] = df[column].round(decimals)

        if drop_original:
            df = df.drop(columns=[column])

        return df

    @staticmethod
    def plot(
        df: pd.DataFrame,
        columns=None,
        mode: str = "lines",
        traces_names=None,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
    ):
        """Plot selected DataFrame columns with Plotly."""
        if not columns:
            columns = df.select_dtypes(include="number").columns.tolist()

        fig = go.Figure()

        for col_idx, col in enumerate(columns):
            trace_name = traces_names[col_idx] if traces_names is not None else col
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col],
                    name=trace_name,
                    connectgaps=False,
                    mode=mode,
                )
            )

        fig.update_layout(
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            template="plotly_white",
            legend_title="Variables",
            title=title,
        )

        return fig

    @staticmethod
    def columns_to_numeric(df: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
        """Convert DataFrame column names to numeric where possible."""
        if copy:
            df = df.copy()

        def to_numeric_safe(value):
            try:
                return pd.to_numeric(value)
            except (ValueError, TypeError):
                return value

        df.columns = [to_numeric_safe(c) for c in df.columns]
        return df


def df_col_to_numeric(col):
    """Compatibility wrapper for ``SeriesUtils.to_numeric()``."""
    return SeriesUtils.to_numeric(col)


def series_fill_gaps(col, max_gap, method="interpolate", value=None):
    """Compatibility wrapper for ``SeriesUtils.fill_gaps()``."""
    return SeriesUtils.fill_gaps(col, max_gap=max_gap, method=method, value=value)


def series_recycle_gaps(s, period="daily", method="resample"):
    """Compatibility wrapper for ``SeriesUtils.recycle_gaps()``."""
    return SeriesUtils.recycle_gaps(s, period=period, method=method)


def df_duplicated_index_counts(df):
    """Compatibility wrapper for ``DataFrameUtils.duplicated_index_counts()``."""
    return DataFrameUtils.duplicated_index_counts(df)


def df_nan_summary(df):
    """Compatibility wrapper for ``DataFrameUtils.nan_summary()``."""
    return DataFrameUtils.nan_summary(df)


def df_check_duplicates(df, name=None, show=True):
    """Compatibility wrapper for ``DataFrameUtils.check_duplicates()``."""
    return DataFrameUtils.check_duplicates(df, name=name, show=show)


def df_time_index_summary(df, plot=False):
    """Compatibility wrapper for ``DataFrameUtils.time_index_summary()``."""
    return DataFrameUtils.time_index_summary(df, plot=plot)


def df_resample(
    df,
    freq,
    method="linear",
    limit_direction="both",
    plot=False,
    plot_column="",
):
    """Compatibility wrapper for ``DataFrameUtils.resample()``."""
    return DataFrameUtils.resample(
        df,
        freq=freq,
        method=method,
        limit_direction=limit_direction,
        plot=plot,
        plot_column=plot_column,
    )


def df_smart_resample(
    df,
    target_freq,
    interpolate_method="linear",
    limit_direction="both",
):
    """Compatibility wrapper for ``DataFrameUtils.smart_resample()``."""
    return DataFrameUtils.smart_resample(
        df,
        target_freq=target_freq,
        interpolate_method=interpolate_method,
        limit_direction=limit_direction,
    )


def df_detect_frequency(index):
    """Compatibility wrapper for ``DateTimeIndexUtils.detect_frequency()``."""
    return DateTimeIndexUtils.detect_frequency(index)


def df_reindex_to_hourly(df):
    """Compatibility wrapper for ``DataFrameUtils.reindex_to_hourly()``."""
    return DataFrameUtils.reindex_to_hourly(df)


def df_duplicate_year(df, start_year, end_year):
    """Compatibility wrapper for ``DataFrameUtils.duplicate_year()``."""
    return DataFrameUtils.duplicate_year(df, start_year=start_year, end_year=end_year)


def df_compute_volume(df, discharge_column="Q", volume_column=None, cumsum_column=None):
    """Compatibility wrapper for ``DataFrameUtils.compute_volume()``."""
    return DataFrameUtils.compute_volume(
        df,
        discharge_column=discharge_column,
        volume_column=volume_column,
        cumsum_column=cumsum_column,
    )


def df_split_column_by_distribution(
    df,
    column,
    percentages,
    new_names,
    drop_original=False,
    decimals=2,
):
    """Compatibility wrapper for ``DataFrameUtils.split_column_by_distribution()``."""
    return DataFrameUtils.split_column_by_distribution(
        df,
        column=column,
        percentages=percentages,
        new_names=new_names,
        drop_original=drop_original,
        decimals=decimals,
    )


def df_split_column_equally(
    df,
    column,
    new_names,
    drop_original=False,
    decimals=4,
):
    """Compatibility wrapper for ``DataFrameUtils.split_column_equally()``."""
    return DataFrameUtils.split_column_equally(
        df,
        column=column,
        new_names=new_names,
        drop_original=drop_original,
        decimals=decimals,
    )


def df_plot(
    df,
    columns=None,
    mode="lines",
    traces_names=None,
    title="",
    xlabel="",
    ylabel="",
):
    """Compatibility wrapper for ``DataFrameUtils.plot()``."""
    return DataFrameUtils.plot(
        df,
        columns=columns,
        mode=mode,
        traces_names=traces_names,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
    )


def df_columns_to_numeric(df, copy=True):
    """Compatibility wrapper for ``DataFrameUtils.columns_to_numeric()``."""
    return DataFrameUtils.columns_to_numeric(df, copy=copy)
