"""Validated writers for Simstrat input tables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path
import stat
import tempfile
from typing import Any

import pandas as pd

from kalden.core.datascience.validation import parse_finite_float

PathLike = str | Path
FlowDefinition = Mapping[str, Any]

__all__ = [
    "generate_inflow_content",
    "inputs_generate_content",
    "write_inflow_file",
]


def _validate_number_format(number_format: str, name: str) -> str:
    if not isinstance(number_format, str) or not number_format:
        raise ValueError(f"{name} must be a non-empty format specification.")
    try:
        format(1.2345, number_format)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {name}: {number_format!r}") from exc
    return number_format


def _parse_flow_definitions(
    frame: pd.DataFrame,
    definitions: Sequence[FlowDefinition],
    flow_type: str,
) -> list[tuple[object, float, str]]:
    parsed: list[tuple[object, float, str]] = []
    for position, definition in enumerate(definitions):
        if not isinstance(definition, Mapping):
            raise TypeError(
                f"{flow_type} flow {position} must be a mapping, "
                f"got {type(definition).__name__}."
            )
        missing_keys = [
            key for key in ("col", "depth") if key not in definition
        ]
        if missing_keys:
            raise ValueError(
                f"{flow_type} flow {position} is missing keys: {missing_keys}."
            )

        column = definition["col"]
        if column not in frame.columns:
            raise KeyError(
                f"{flow_type} flow column {column!r} is not in the dataframe."
            )
        depth = parse_finite_float(definition["depth"])
        if depth is None:
            raise ValueError(
                f"{flow_type} flow {position} has an invalid depth: "
                f"{definition['depth']!r}."
            )
        header = str(definition.get("header", column)).strip()
        if not header:
            raise ValueError(f"{flow_type} flow {position} has an empty header.")
        parsed.append((column, depth, header))
    return parsed


def _validate_datetime_index(frame: pd.DataFrame) -> None:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("frame.index must be a pandas DatetimeIndex.")
    if frame.index.hasnans:
        raise ValueError("frame.index contains invalid timestamps.")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("frame.index must be sorted chronologically.")
    if frame.index.has_duplicates:
        raise ValueError("frame.index contains duplicate timestamps.")
    if frame.columns.has_duplicates:
        raise ValueError("frame.columns contains duplicate labels.")


def _format_value(
    value: object,
    *,
    number_format: str,
    allow_missing: bool,
    missing_value: str,
    row_label: object,
    column: object,
) -> str:
    if pd.isna(value):
        if allow_missing:
            return missing_value
        raise ValueError(
            f"Missing value at timestamp {row_label!r}, column {column!r}."
        )
    number = parse_finite_float(value)
    if number is None:
        raise ValueError(
            f"Invalid numeric value at timestamp {row_label!r}, "
            f"column {column!r}: {value!r}."
        )
    return format(number, number_format)


def generate_inflow_content(
    frame: pd.DataFrame,
    reference_date: str | pd.Timestamp,
    deep_flows: Sequence[FlowDefinition],
    surface_flows: Sequence[FlowDefinition],
    sep: str = "\t",
    headers: str = "",
    *,
    value_format: str = ".10g",
    time_format: str = ".10g",
    allow_missing: bool = False,
    missing_value: str = "",
    drop_before_reference: bool = True,
) -> str:
    """Generate a validated Simstrat inflow table without quantizing values."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    if not isinstance(sep, str) or not sep:
        raise ValueError("sep must be a non-empty string.")
    value_format = _validate_number_format(value_format, "value_format")
    time_format = _validate_number_format(time_format, "time_format")
    _validate_datetime_index(frame)

    reference = pd.Timestamp(reference_date)
    if pd.isna(reference):
        raise ValueError("reference_date must be a valid timestamp.")
    if frame.index.tz != reference.tz:
        raise ValueError(
            "frame.index and reference_date must use the same timezone convention."
        )

    deep = _parse_flow_definitions(frame, deep_flows, "deep")
    surface = _parse_flow_definitions(frame, surface_flows, "surface")
    flows = [*deep, *surface]
    if not flows:
        raise ValueError("At least one deep or surface flow is required.")

    days = pd.Series(
        (frame.index - reference).total_seconds() / 86_400,
        index=frame.index,
        name="Time [d]",
    )
    selected = frame
    if drop_before_reference:
        keep = days >= 0
        selected = frame.loc[keep]
        days = days.loc[keep]
    if selected.empty:
        raise ValueError("No rows remain on or after the reference date.")

    if headers:
        lines = headers.rstrip("\r\n").splitlines()
    else:
        lines = [sep + sep.join(["Time [d]", *(item[2] for item in flows)])]
    lines.append(sep + sep.join([str(len(deep)), str(len(surface))]))
    depth_values = [format(item[1], value_format) for item in flows]
    lines.append("-1" + sep + sep + sep.join(depth_values))

    for timestamp, day in days.items():
        row = [format(float(day), time_format)]
        row.extend(
            _format_value(
                selected.at[timestamp, column],
                number_format=value_format,
                allow_missing=allow_missing,
                missing_value=missing_value,
                row_label=timestamp,
                column=column,
            )
            for column, _, _ in flows
        )
        lines.append(sep + sep.join(row))
    return "\n".join(lines) + "\n"


def inputs_generate_content(
    df: pd.DataFrame,
    ref_date: str | pd.Timestamp,
    deep_flows: Sequence[FlowDefinition],
    surface_flows: Sequence[FlowDefinition],
    sep: str = "\t",
    headers: str = "",
    **kwargs: Any,
) -> str:
    """Compatibility wrapper for the legacy notebook writer name."""
    return generate_inflow_content(
        df,
        ref_date,
        deep_flows,
        surface_flows,
        sep=sep,
        headers=headers,
        **kwargs,
    )


def write_inflow_file(
    path: PathLike,
    frame: pd.DataFrame,
    reference_date: str | pd.Timestamp,
    deep_flows: Sequence[FlowDefinition],
    surface_flows: Sequence[FlowDefinition],
    sep: str = "\t",
    headers: str = "",
    *,
    atomic: bool = True,
    encoding: str = "utf-8",
    **kwargs: Any,
) -> Path:
    """Write a Simstrat table to its live path, atomically by default."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    content = generate_inflow_content(
        frame,
        reference_date,
        deep_flows,
        surface_flows,
        sep=sep,
        headers=headers,
        **kwargs,
    )

    if not atomic:
        target.write_text(content, encoding=encoding, newline="\n")
        return target.resolve()

    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding=encoding, newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if target.exists():
            os.chmod(temporary, stat.S_IMODE(target.stat().st_mode))
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target.resolve()
