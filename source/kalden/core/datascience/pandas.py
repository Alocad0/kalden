"""Generic pandas helpers used across projects.

The module is organized by pandas object type:

* ``SeriesUtils`` for ``pandas.Series`` helpers
* ``DateTimeIndexUtils`` for ``pandas.DatetimeIndex`` helpers
* ``DataFrameUtils`` for ``pandas.DataFrame`` helpers

Thin function wrappers are kept at module level for convenience and
compatibility with earlier utility-style usage.
"""

from __future__ import annotations

import re
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
    def convert_numeric_like_columns(
        df: pd.DataFrame,
        exclude: list[str] | None = None,
        copy: bool = True,
        normalize_numeric_strings: bool = False,
        decimal_separator: str = "mixed",
    ) -> pd.DataFrame:
        """Convert object/string columns to float when all non-null/non-empty
        values in the column can be converted to numeric.
    
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to process.
        exclude : list[str], optional
            Columns to leave untouched.
        copy : bool, default True
            Whether to return a copy.
        normalize_numeric_strings : bool, default False
            If True, normalize numeric strings before conversion. This allows
            values using comma or dot decimal separators, as well as common
            thousands separators.
        decimal_separator : {".", ",", "mixed"}, default "mixed"
            Decimal separator strategy used when normalize_numeric_strings=True.
            - "." assumes dot decimal and comma thousands.
            - "," assumes comma decimal and dot thousands.
            - "mixed" infers the decimal separator per value.
        """
        if copy:
            df = df.copy()
    
        if exclude is None:
            exclude = []
    
        for col in df.columns:
            if col in exclude:
                continue
    
            if not (
                pd.api.types.is_object_dtype(df[col])
                or pd.api.types.is_string_dtype(df[col])
            ):
                continue
    
            cleaned = df[col].replace(r"^\s*$", pd.NA, regex=True)
    
            if normalize_numeric_strings:
                numeric_input = cleaned.map(
                    lambda value: DataFrameUtils._normalize_numeric_value(
                        value,
                        decimal_separator=decimal_separator,
                    )
                )
            else:
                numeric_input = cleaned
    
            converted = pd.to_numeric(numeric_input, errors="coerce")
            had_value = cleaned.notna()
    
            if converted[had_value].notna().all():
                df[col] = converted.astype(float)
    
        return df

    @staticmethod
    def _normalize_numeric_value(value, decimal_separator: str = "mixed"):
        """Normalize numeric strings before pd.to_numeric.
    
        Supports:
        - 23.2
        - 23,2
        - 1,234.56
        - 1.234,56
        - 1 234,56
        - 1'234.56
    
        Parameters
        ----------
        value : object
            Cell value.
        decimal_separator : {".", ",", "mixed"}, default "mixed"
            Decimal separator strategy.
            - "." assumes dot decimal and comma thousands.
            - "," assumes comma decimal and dot thousands.
            - "mixed" infers per value.
        """
        if pd.isna(value):
            return pd.NA
    
        if isinstance(value, (int, float, np.number)):
            return value
    
        text = str(value).strip()
    
        if text == "":
            return pd.NA
    
        # Remove common thousands spacing separators
        text = (
            text.replace("\u00a0", "")
            .replace("\u202f", "")
            .replace(" ", "")
            .replace("'", "")
            .replace("’", "")
        )
    
        # Support accounting-style negative values: (123,45)
        if text.startswith("(") and text.endswith(")"):
            text = "-" + text[1:-1]
    
        match = re.fullmatch(r"([+-]?)(.*?)([eE][+-]?\d+)?", text)
    
        if not match:
            return text
    
        sign, mantissa, exponent = match.groups()
        exponent = exponent or ""
    
        if not re.fullmatch(r"\d*([,.]\d*)*", mantissa):
            return text
    
        def looks_like_grouped_integer(number: str, sep: str) -> bool:
            parts = number.split(sep)
            return (
                len(parts) > 1
                and parts[0].isdigit()
                and 1 <= len(parts[0]) <= 3
                and all(part.isdigit() and len(part) == 3 for part in parts[1:])
            )
    
        has_comma = "," in mantissa
        has_dot = "." in mantissa
    
        if decimal_separator not in {".", ",", "mixed"}:
            raise ValueError('decimal_separator must be ".", "," or "mixed".')
    
        if decimal_separator == "mixed":
            if has_comma and has_dot:
                # The rightmost separator is usually the decimal mark:
                # 1,234.56 -> dot
                # 1.234,56 -> comma
                decimal_mark = "," if mantissa.rfind(",") > mantissa.rfind(".") else "."
            elif has_comma:
                # 23,2 -> decimal comma
                # 1,234,567 -> thousands grouping
                decimal_mark = None if looks_like_grouped_integer(mantissa, ",") else ","
            elif has_dot:
                # 23.2 -> decimal dot
                # 1.234.567 -> thousands grouping
                decimal_mark = None if looks_like_grouped_integer(mantissa, ".") else "."
            else:
                decimal_mark = "."
        else:
            decimal_mark = decimal_separator
    
        if decimal_mark is None:
            mantissa = mantissa.replace(",", "").replace(".", "")
        elif decimal_mark == ",":
            mantissa = mantissa.replace(".", "")
            mantissa = mantissa.replace(",", ".")
        else:
            mantissa = mantissa.replace(",", "")
    
        return f"{sign}{mantissa}{exponent}"

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
        discharge_column: str | list[str] | None = "Q",
        unit: str = "m3/s",
        method: str = "step",
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Compute timestep, cumulative, and total volumes from discharge time series.

        This method integrates one or more discharge columns over time using the
        DataFrame's DatetimeIndex. It returns both a detailed volume DataFrame and
        the total integrated volume for each discharge column.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing discharge time series.

            The DataFrame must have a pandas DatetimeIndex. The index represents
            the timestamps of the discharge values and is sorted internally before
            integration.

        discharge_column : str, list of str, or None, default "Q"
            Column or columns containing discharge values.

            If a string is provided, only that column is integrated.

            If a list of strings is provided, all listed columns are integrated.

            If None is provided, all columns in the DataFrame are integrated.

        unit : {"m3/s", "l/s"}, default "m3/s"
            Unit of the discharge values.

            - "m3/s" means cubic metres per second.
            - "l/s" means litres per second.

            When `unit="l/s"`, values are converted internally to m³/s before
            integration.

        method : {"step", "trapezoidal"}, default "step"
            Integration method.

            - "step"
                Assumes each discharge value remains constant until the next
                timestamp.

                The volume assigned to timestamp `t_i` is calculated over the
                interval `[t_i, t_{i+1})`.

                The final timestamp contributes zero volume because no following
                timestamp is available.

            - "trapezoidal"
                Assumes discharge varies linearly between consecutive timestamps.

                The volume assigned to timestamp `t_i` is calculated over the
                interval `(t_{i-1}, t_i]`, using the mean discharge between
                `t_{i-1}` and `t_i`.

                The first timestamp contributes zero volume because no previous
                timestamp is available.

        Returns
        -------
        volume_df : pd.DataFrame
            DataFrame with the same DatetimeIndex as the input.

            The returned DataFrame has a two-level column index:

            - "volume_m3"
                Volume per timestep, in m³.

            - "cumulative_volume_m3"
                Accumulated volume up to and including each timestep, in m³.

            The second column level contains the original discharge column names.

        total_volume : pd.Series
            Total integrated volume per discharge column, in m³.

            This is equivalent to:

            ```
            volume_df["volume_m3"].sum()
            ```

            or:

            ```
            volume_df["cumulative_volume_m3"].iloc[-1]
            ```

        Raises
        ------
        TypeError
            If `df` is not a pandas DataFrame.
            If `df.index` is not a pandas DatetimeIndex.

        ValueError
            If `unit` is not "m3/s" or "l/s".
            If `method` is not "step" or "trapezoidal".
            If one or more requested discharge columns are missing.

        Notes
        -----
        This method does not extrapolate beyond the provided timestamps.

        For the "step" method, the final discharge value is not assigned a
        volume unless an explicit following timestamp exists. If the final value
        should apply over a known duration, append an explicit final timestamp
        before calling this method.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("df index must be a pandas DatetimeIndex.")

        if unit not in ["m3/s", "l/s"]:
            raise ValueError('unit must be "m3/s" or "l/s".')

        if method not in ["step", "trapezoidal"]:
            raise ValueError('method must be "step" or "trapezoidal".')

        df = df.sort_index().copy()

        if discharge_column is None:
            discharge = df.copy()
        else:
            if isinstance(discharge_column, str):
                discharge_columns = [discharge_column]
            else:
                discharge_columns = list(discharge_column)

            missing_columns = [
                column for column in discharge_columns if column not in df.columns
            ]

            if missing_columns:
                raise ValueError(
                    "The following discharge columns are missing from df: "
                    f"{missing_columns}"
                )

            discharge = df[discharge_columns].copy()

        if unit == "l/s":
            discharge = discharge / 1000

        if method == "step":
            dt_seconds = (
                discharge.index
                .to_series()
                .diff()
                .dt.total_seconds()
                .shift(-1)
                .fillna(0)
            )

            volume_per_timestep = discharge.multiply(dt_seconds, axis=0)

        else:
            dt_seconds = (
                discharge.index
                .to_series()
                .diff()
                .dt.total_seconds()
                .fillna(0)
            )

            mean_discharge = (discharge + discharge.shift(1)) / 2
            volume_per_timestep = (
                mean_discharge
                .multiply(dt_seconds, axis=0)
                .fillna(0)
            )

        cumulative_volume = volume_per_timestep.cumsum()
        total_volume = volume_per_timestep.sum()

        volume_df = pd.concat(
            {
                "volume_m3": volume_per_timestep,
                "cumulative_volume_m3": cumulative_volume,
            },
            axis=1,
        )

        return volume_df, total_volume

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

def df_convert_numeric_like_columns(
    df,
    exclude=None,
    copy=True,
    normalize_numeric_strings=False,
    decimal_separator="mixed",
):
    """Compatibility wrapper for DataFrameUtils.convert_numeric_like_columns()."""
    return DataFrameUtils.convert_numeric_like_columns(
        df,
        exclude=exclude,
        copy=copy,
        normalize_numeric_strings=normalize_numeric_strings,
        decimal_separator=decimal_separator,
    )

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


def df_compute_volume(
    df,
    discharge_column="Q",
    unit="m3/s",
    method="step",
):
    """Compatibility wrapper for ``DataFrameUtils.compute_volume()``."""
    return DataFrameUtils.compute_volume(
        df,
        discharge_column=discharge_column,
        unit=unit,
        method=method,
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
