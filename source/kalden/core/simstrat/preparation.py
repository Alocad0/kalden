"""Data preparation helpers for Simstrat model inputs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

from kalden.core.datascience.validation import parse_finite_float

MissingMode = Literal["raise", "keep"]

__all__ = [
    "mgL_to_mmolm3",
    "mg_l_to_mmol_m3",
    "sum_complete_flows",
]


def sum_complete_flows(
    frame: pd.DataFrame,
    columns: Sequence[object] | None = None,
    *,
    name: str = "Total",
    missing: MissingMode = "raise",
) -> pd.Series:
    """Sum flow columns without silently treating missing values as zero.

    Resample or otherwise align the individual flow series before calling this
    function. With the default policy, any incomplete timestamp raises and
    identifies the first gap. ``missing="keep"`` retains incomplete rows as
    ``NaN`` for an explicit downstream imputation step.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    if missing not in {"raise", "keep"}:
        raise ValueError("missing must be either 'raise' or 'keep'.")

    selected_columns = list(frame.columns if columns is None else columns)
    if not selected_columns:
        raise ValueError("At least one flow column is required.")

    absent = [column for column in selected_columns if column not in frame.columns]
    if absent:
        raise KeyError(f"Flow columns not found: {absent}")

    numeric = frame.loc[:, selected_columns].apply(
        pd.to_numeric,
        errors="raise",
    )
    total = numeric.sum(axis=1, min_count=len(selected_columns))
    total.name = name

    incomplete = total.isna()
    if incomplete.any() and missing == "raise":
        first = total.index[incomplete][0]
        raise ValueError(
            f"Cannot compute {name!r}: {int(incomplete.sum())} timestamps have "
            f"missing flow components; first incomplete timestamp: {first}."
        )
    return total


def mg_l_to_mmol_m3(
    values: float | Sequence[float] | pd.Series,
    molar_mass_g_mol: float,
) -> float | np.ndarray | pd.Series:
    """Convert mg/L to mmol/m3 without applying display rounding."""
    molar_mass = parse_finite_float(molar_mass_g_mol)
    if molar_mass is None or molar_mass <= 0:
        raise ValueError("molar_mass_g_mol must be a positive finite number.")

    factor = 1000.0 / molar_mass
    if isinstance(values, pd.Series):
        numeric = pd.to_numeric(values, errors="raise")
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError("Concentrations must all be finite numbers.")
        return numeric * factor

    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("Concentrations must all be finite numbers.")
    converted = array * factor
    if array.ndim == 0:
        return float(converted)
    return converted


def mgL_to_mmolm3(
    values: float | Sequence[float] | pd.Series,
    molar_mass_g_mol: float,
) -> float | np.ndarray | pd.Series:
    """Compatibility alias for :func:`mg_l_to_mmol_m3`."""
    return mg_l_to_mmol_m3(values, molar_mass_g_mol)
