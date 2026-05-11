"""Notebook-friendly utilities to convert CSV files to MIKE DFS0 from an Excel config.

This module is a simplified rewrite of the uploaded script:
- it avoids ``Path.resolve()`` and similar path normalization that can be fragile
  on some server setups
- it uses plain ``os.path`` and string paths
- it is designed to be imported and called directly from a notebook

Expected workbook sheets
------------------------
1) Jobs
2) Items

Required packages
-----------------
pip install pandas openpyxl mikeio

Typical notebook usage
----------------------
>>> import csv_to_dfs0_module as c2d
>>> completed = c2d.run("csv_to_dfs0_config.xlsx")
>>> completed = c2d.run("csv_to_dfs0_config.xlsx", selected_jobs={"example_01"})
>>> ds, df = c2d.build_dataset_for_job(job, items_df)

Notes
-----
- Paths can be absolute or relative to the Excel config workbook.
- Non-equidistant timestamps are supported as long as the time column is parseable.
- If ``output_dfs0_path`` is blank, the output is written next to the CSV file with
  the same filename stem and a ``.dfs0`` extension.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterable

import pandas as pd


REQUIRED_JOB_COLUMNS = {
    "job_id",
    "enabled",
    "csv_path",
    "output_dfs0_path",
    "time_column",
    "delimiter",
    "decimal",
    "encoding",
    "skiprows",
    "header_row",
    "na_values",
    "dayfirst",
    "time_format",
    "drop_rows_all_nan",
}

REQUIRED_ITEM_COLUMNS = {
    "job_id",
    "order",
    "enabled",
    "csv_column",
    "item_name",
    "eum_type",
    "eum_unit",
    "data_value_type",
    "scale_factor",
    "offset",
}


__all__ = [
    "Job",
    "REQUIRED_JOB_COLUMNS",
    "REQUIRED_ITEM_COLUMNS",
    "load_config",
    "iter_jobs",
    "list_enabled_jobs",
    "build_dataset_for_job",
    "write_job",
    "run",
    "run_one",
]


@dataclass
class Job:
    """Configuration for one CSV-to-DFS0 conversion job.

    Parameters
    ----------
    job_id : str
        Unique job identifier.
    csv_path : str
        Path to the source CSV file.
    output_path : str
        Path to the output DFS0 file.
    time_column : str
        Name of the CSV column containing timestamps.
    delimiter : str, default ","
        CSV delimiter.
    decimal : str, default "."
        Decimal separator used in the CSV file.
    encoding : str | None, default None
        Optional CSV encoding.
    skiprows : int, default 0
        Number of rows to skip before reading the CSV data.
    header_row : int, default 0
        Row index to use as the header in ``pandas.read_csv``.
    na_values : list[str] | None, default None
        Optional list of extra missing-value markers.
    dayfirst : bool, default False
        Forwarded to ``pandas.to_datetime``.
    time_format : str | None, default None
        Optional explicit timestamp format.
    drop_rows_all_nan : bool, default True
        Drop rows where all data columns are NaN before building the dataset.
    """

    job_id: str
    csv_path: str
    output_path: str
    time_column: str
    delimiter: str = ","
    decimal: str = "."
    encoding: str | None = None
    skiprows: int = 0
    header_row: int = 0
    na_values: list[str] | None = None
    dayfirst: bool = False
    time_format: str | None = None
    drop_rows_all_nan: bool = True


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of a configuration dataframe.

    This trims whitespace from column names and string cells and removes rows that
    are completely empty.
    """
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    if df.empty:
        return df

    for col in df.columns:
        df[col] = df[col].map(lambda value: value.strip() if isinstance(value, str) else value)

    return df.dropna(how="all")


def _truthy(value) -> bool:
    """Interpret common spreadsheet truthy values as ``True``."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"y", "yes", "true", "1", "on"}


def _blank_to_none(value):
    """Convert empty spreadsheet-style values to ``None``."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    return None if text == "" else text


def _to_int(value, default: int = 0) -> int:
    """Convert a spreadsheet cell to ``int`` with a fallback default."""
    text = _blank_to_none(value)
    if text is None:
        return default
    return int(float(text))


def _to_float(value, default: float = 0.0) -> float:
    """Convert a spreadsheet cell to ``float`` with a fallback default."""
    text = _blank_to_none(value)
    if text is None:
        return default
    return float(text)


def _pipe_split(value) -> list[str] | None:
    """Split a pipe-delimited config value into a list of strings."""
    text = _blank_to_none(value)
    if text is None:
        return None
    return [part.strip() for part in str(text).split("|") if part.strip()]


def _config_dir(config_path: str) -> str:
    """Return the directory of the Excel config file using plain ``os.path``.

    No ``Path.resolve()`` or similar normalization is used here.
    """
    config_text = os.fspath(config_path)
    directory = os.path.dirname(config_text)
    return directory if directory else "."


def _path_from_config(value: str | None, config_dir: str, default_suffix: str | None = None) -> str | None:
    """Build a path relative to the config workbook directory.

    Parameters
    ----------
    value : str | None
        Raw path value from the Excel workbook.
    config_dir : str
        Directory containing the Excel workbook.
    default_suffix : str | None, optional
        Suffix to append when the provided path has no extension.

    Returns
    -------
    str | None
        A normalized string path, or ``None`` when the input is blank.
    """
    text = _blank_to_none(value)
    if text is None:
        return None

    if os.path.isabs(text):
        path = text
    else:
        path = os.path.join(config_dir, text)

    path = os.path.normpath(path)

    if default_suffix and os.path.splitext(path)[1] == "":
        path = path + default_suffix

    return path


def _normalize_data_value_type(value: str | None) -> str:
    """Normalize common textual variants of MIKE data value types."""
    text = _blank_to_none(value)
    if text is None:
        return "Instantaneous"

    key = str(text).strip().replace("-", " ").replace("_", " ").lower()

    mapping = {
        "instantaneous": "Instantaneous",
        "accumulated": "Accumulated",
        "step accumulated": "StepAccumulated",
        "stepaccumulated": "StepAccumulated",
        "mean step backward": "MeanStepBackward",
        "meanstepbackward": "MeanStepBackward",
        "mean step forward": "MeanStepForward",
        "meanstepforward": "MeanStepForward",
    }
    return mapping.get(key, str(text).strip())


def _import_mikeio():
    """Import ``mikeio`` with a clear error message if unavailable."""
    try:
        import mikeio  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Missing dependency: mikeio\n"
            "Install it with:\n"
            "    pip install mikeio\n"
        ) from exc
    return mikeio


def load_config(config_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the Excel configuration workbook.

    Parameters
    ----------
    config_path : str
        Path to the Excel workbook.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        The cleaned ``Jobs`` and ``Items`` sheets.
    """
    jobs_df = pd.read_excel(config_path, sheet_name="Jobs")
    items_df = pd.read_excel(config_path, sheet_name="Items")

    jobs_df = _clean_frame(jobs_df)
    items_df = _clean_frame(items_df)

    missing_jobs = REQUIRED_JOB_COLUMNS - set(jobs_df.columns)
    missing_items = REQUIRED_ITEM_COLUMNS - set(items_df.columns)

    if missing_jobs:
        raise ValueError(
            "Missing required columns in Jobs sheet: "
            f"{sorted(missing_jobs)}"
        )
    if missing_items:
        raise ValueError(
            "Missing required columns in Items sheet: "
            f"{sorted(missing_items)}"
        )

    return jobs_df, items_df


def iter_jobs(
    jobs_df: pd.DataFrame,
    config_path: str,
    job_filter: set[str] | None = None,
) -> Iterable[Job]:
    """Yield enabled jobs from the Jobs sheet.

    Parameters
    ----------
    jobs_df : pandas.DataFrame
        Cleaned Jobs sheet.
    config_path : str
        Path to the Excel workbook.
    job_filter : set[str] | None, optional
        Optional subset of job IDs to keep.
    """
    config_dir = _config_dir(config_path)

    for _, row in jobs_df.iterrows():
        if not _truthy(row["enabled"]):
            continue

        job_id = str(row["job_id"]).strip()
        if not job_id:
            continue
        if job_filter and job_id not in job_filter:
            continue

        csv_path = _path_from_config(row["csv_path"], config_dir)
        if csv_path is None:
            raise ValueError(f"Job '{job_id}' is missing csv_path.")

        output_path = _path_from_config(row["output_dfs0_path"], config_dir)
        if output_path is None:
            output_path = os.path.splitext(csv_path)[0] + ".dfs0"

        yield Job(
            job_id=job_id,
            csv_path=csv_path,
            output_path=output_path,
            time_column=str(row["time_column"]).strip(),
            delimiter=_blank_to_none(row["delimiter"]) or ",",
            decimal=_blank_to_none(row["decimal"]) or ".",
            encoding=_blank_to_none(row["encoding"]),
            skiprows=_to_int(row["skiprows"], 0),
            header_row=_to_int(row["header_row"], 0),
            na_values=_pipe_split(row["na_values"]),
            dayfirst=_truthy(row["dayfirst"]),
            time_format=_blank_to_none(row["time_format"]),
            drop_rows_all_nan=(
                _truthy(row["drop_rows_all_nan"])
                if _blank_to_none(row["drop_rows_all_nan"]) is not None
                else True
            ),
        )


def list_enabled_jobs(config_path: str) -> list[str]:
    """Return the list of enabled job IDs from the workbook."""
    jobs_df, _ = load_config(config_path)
    return [job.job_id for job in iter_jobs(jobs_df, config_path)]


def _make_item_info_factory():
    """Return a helper that builds ``mikeio.ItemInfo`` objects.

    The helper keeps a few constructor fallbacks so it remains usable across MIKE IO
    variants that differ slightly in ``ItemInfo`` signatures.
    """
    mikeio = _import_mikeio()
    ItemInfo = mikeio.ItemInfo
    EUMType = mikeio.EUMType
    EUMUnit = mikeio.EUMUnit

    def make_item_info(
        item_name: str,
        eum_type_name: str,
        eum_unit_name: str | None,
        data_value_type: str,
    ):
        try:
            eum_type = getattr(EUMType, eum_type_name)
        except AttributeError as exc:
            raise ValueError(
                f"Unknown EUM type '{eum_type_name}'. "
                "Tip: in Python, try EUMType.search('keyword') to discover valid names."
            ) from exc

        eum_unit = None
        if eum_unit_name:
            try:
                eum_unit = getattr(EUMUnit, eum_unit_name)
            except AttributeError as exc:
                raise ValueError(
                    f"Unknown EUM unit '{eum_unit_name}'. "
                    f"Tip: valid units for {eum_type_name} can often be inspected with "
                    f"EUMType.{eum_type_name}.units"
                ) from exc

        candidates = []
        if eum_unit is not None:
            candidates.extend(
                [
                    lambda: ItemInfo(item_name, eum_type, eum_unit, data_value_type=data_value_type),
                    lambda: ItemInfo(item_name, eum_type, eum_unit, data_value_type),
                    lambda: ItemInfo(eum_type, eum_unit, data_value_type=data_value_type),
                    lambda: ItemInfo(eum_type, eum_unit, data_value_type),
                ]
            )
        else:
            candidates.extend(
                [
                    lambda: ItemInfo(item_name, eum_type, data_value_type=data_value_type),
                    lambda: ItemInfo(item_name, eum_type, data_value_type),
                    lambda: ItemInfo(eum_type, data_value_type=data_value_type),
                    lambda: ItemInfo(eum_type, data_value_type),
                ]
            )

        last_error = None
        info = None
        for fn in candidates:
            try:
                info = fn()
                break
            except TypeError as exc:
                last_error = exc

        if info is None:
            raise TypeError(
                f"Could not construct ItemInfo for '{item_name}'. "
                f"Last error: {last_error}"
            )

        if getattr(info, "name", None) != item_name:
            try:
                info.name = item_name
            except Exception:
                pass

        return info

    return make_item_info


def build_dataset_for_job(job: Job, items_df: pd.DataFrame):
    """Build a ``mikeio.Dataset`` and its source DataFrame for one job.

    Parameters
    ----------
    job : Job
        One conversion job.
    items_df : pandas.DataFrame
        Cleaned Items sheet.

    Returns
    -------
    tuple[mikeio.Dataset, pandas.DataFrame]
        The DFS0-ready dataset and the numeric time-indexed DataFrame used to
        build it.
    """
    mikeio = _import_mikeio()
    make_item_info = _make_item_info_factory()

    item_rows = items_df.copy()
    item_rows = item_rows[item_rows["job_id"].astype(str).str.strip() == job.job_id]
    item_rows = item_rows[item_rows["enabled"].map(_truthy)]

    if item_rows.empty:
        raise ValueError(f"No enabled item rows found for job '{job.job_id}'.")

    item_rows = item_rows.assign(_sort_order=item_rows["order"].map(lambda value: _to_int(value, 999999)))
    item_rows = item_rows.sort_values(["_sort_order", "csv_column"])

    csv_kwargs = {
        "sep": job.delimiter,
        "decimal": job.decimal,
        "skiprows": job.skiprows,
        "header": job.header_row,
    }
    if job.encoding:
        csv_kwargs["encoding"] = job.encoding
    if job.na_values:
        csv_kwargs["na_values"] = job.na_values

    if not os.path.isfile(job.csv_path):
        raise FileNotFoundError(f"CSV file not found: {job.csv_path}")

    df = pd.read_csv(job.csv_path, **csv_kwargs)

    if job.time_column not in df.columns:
        raise ValueError(
            f"Time column '{job.time_column}' was not found in {job.csv_path}. "
            f"Columns found: {list(df.columns)}"
        )

    if job.time_format:
        time_index = pd.to_datetime(
            df[job.time_column],
            format=job.time_format,
            dayfirst=job.dayfirst,
            errors="coerce",
        )
    else:
        time_index = pd.to_datetime(
            df[job.time_column],
            dayfirst=job.dayfirst,
            errors="coerce",
        )

    bad_time_rows = int(time_index.isna().sum())
    if bad_time_rows:
        raise ValueError(
            f"Job '{job.job_id}' has {bad_time_rows} unparseable timestamps "
            f"in column '{job.time_column}'."
        )

    df = df.copy()
    df.index = pd.DatetimeIndex(time_index, name=job.time_column)
    df = df.drop(columns=[job.time_column])
    df = df.sort_index()

    if df.index.has_duplicates:
        dupes = int(df.index.duplicated().sum())
        raise ValueError(
            f"Job '{job.job_id}' has {dupes} duplicate timestamps after parsing. "
            "Please de-duplicate the CSV before export."
        )

    data = pd.DataFrame(index=df.index)
    item_infos = []

    for _, row in item_rows.iterrows():
        csv_column = str(row["csv_column"]).strip()
        if csv_column not in df.columns:
            raise ValueError(
                f"Job '{job.job_id}' expects CSV column '{csv_column}', "
                f"but it was not found in {job.csv_path}."
            )

        item_name = _blank_to_none(row["item_name"]) or csv_column
        if item_name in data.columns:
            raise ValueError(
                f"Duplicate target item name '{item_name}' in Items sheet for job '{job.job_id}'."
            )

        series = pd.to_numeric(df[csv_column], errors="coerce")
        scale_factor = _to_float(row["scale_factor"], 1.0)
        offset = _to_float(row["offset"], 0.0)
        series = series * scale_factor + offset
        data[item_name] = series

        eum_type_name = str(row["eum_type"]).strip()
        eum_unit_name = _blank_to_none(row["eum_unit"])
        data_value_type = _normalize_data_value_type(row["data_value_type"])

        item_infos.append(
            make_item_info(
                item_name=item_name,
                eum_type_name=eum_type_name,
                eum_unit_name=eum_unit_name,
                data_value_type=data_value_type,
            )
        )

    if job.drop_rows_all_nan:
        data = data.dropna(how="all")

    if data.empty:
        raise ValueError(f"Job '{job.job_id}' produced an empty dataset after filtering.")

    dataset = mikeio.from_pandas(data, items=item_infos)
    return dataset, data


def write_job(job: Job, items_df: pd.DataFrame, dry_run: bool = False):
    """Build and optionally write a DFS0 file for one job.

    Parameters
    ----------
    job : Job
        Job definition.
    items_df : pandas.DataFrame
        Cleaned Items sheet.
    dry_run : bool, default False
        If ``True``, validate everything without writing the DFS0 file.

    Returns
    -------
    tuple[mikeio.Dataset, pandas.DataFrame]
        The built dataset and source DataFrame.
    """
    dataset, data = build_dataset_for_job(job, items_df)

    if not dry_run:
        output_dir = os.path.dirname(job.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        dataset.to_dfs(job.output_path)

    return dataset, data


def run(
    config_path: str,
    selected_jobs: set[str] | None = None,
    dry_run: bool = False,
    verbose: bool = True,
) -> int:
    """Run all enabled jobs from an Excel workbook.

    Parameters
    ----------
    config_path : str
        Path to the Excel workbook.
    selected_jobs : set[str] | None, optional
        Optional subset of job IDs to execute.
    dry_run : bool, default False
        Validate parsing and dataset creation without writing DFS0 files.
    verbose : bool, default True
        Print one block of progress information per job.

    Returns
    -------
    int
        Number of completed jobs.
    """
    jobs_df, items_df = load_config(config_path)
    jobs = list(iter_jobs(jobs_df, config_path, selected_jobs))

    if selected_jobs:
        found = {job.job_id for job in jobs}
        missing = sorted(selected_jobs - found)
        if missing:
            raise ValueError(f"Requested job_id(s) not found or not enabled: {missing}")

    if not jobs:
        raise ValueError("No enabled jobs found in the Jobs sheet.")

    completed = 0
    for job in jobs:
        _, data = write_job(job, items_df, dry_run=dry_run)

        if verbose:
            print(f"[OK] {job.job_id}")
            print(f"     CSV   : {job.csv_path}")
            print(f"     Rows  : {len(data):,}")
            print(f"     Items : {len(data.columns)}")
            print(f"     Time  : {data.index.min()} -> {data.index.max()}")
            print(f"     DFS0  : {job.output_path}")
            if not dry_run:
                print("     Wrote : yes")
            else:
                print("     Wrote : no (dry_run=True)")
            print()

        completed += 1

    return completed


def run_one(config_path: str, job_id: str, dry_run: bool = False, verbose: bool = True) -> int:
    """Run a single job by ID.

    This is a small notebook-friendly convenience wrapper around :func:`run`.
    """
    return run(
        config_path=config_path,
        selected_jobs={job_id},
        dry_run=dry_run,
        verbose=verbose,
    )
