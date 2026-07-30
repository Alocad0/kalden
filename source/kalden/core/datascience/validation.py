"""Small scalar validation helpers.

Place this module at::

    source/kalden/core/datascience/validation.py

``_is_numeric`` is intentionally kept private for compatibility with the
existing helper in ``datascience/pandas.py``. Once this module is added, keep a
single implementation here and import it from ``pandas.py`` where required.
"""

from __future__ import annotations

import math
from typing import Any
from numbers import Real

__all__ = ["_is_numeric", "parse_finite_float"]


from numbers import Real

def _is_numeric(value):
    if isinstance(value, Real):
        return True

    if isinstance(value, (str, bytes)):
        value = value.strip()
        if not value:
            return False

        try:
            int(value, 0)
            return True
        except (TypeError, ValueError):
            pass

        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False

    return False

def _is_boolean(value: Any) -> bool:
    """Return True when value is a supported boolean representation."""
    if isinstance(value, bool):
        return True

    if isinstance(value, Real):
        return value in (0, 1)

    if isinstance(value, str):
        return value.strip().lower() in {"true", "false", "0", "1"}

    return False

def parse_finite_float(value: Any) -> float | None:
    """Parse *value* as a finite float, returning ``None`` when invalid.

    The function accepts ordinary numeric values and numeric strings. Booleans,
    empty strings, NaN, positive/negative infinity, and unsupported objects are
    rejected. Python-style integer strings such as ``"0x10"`` are accepted to
    remain compatible with :func:`_is_numeric`.

    Examples
    --------
    >>> parse_finite_float("1.25")
    1.25
    >>> parse_finite_float("0x10")
    16.0
    >>> parse_finite_float("nan") is None
    True
    >>> parse_finite_float(True) is None
    True
    """
    if isinstance(value, bool):
        return None

    if isinstance(value, (str, bytes)):
        value = value.strip()
        if not value:
            return None

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        if not isinstance(value, (str, bytes)):
            return None

        try:
            number = float(int(value, 0))
        except (TypeError, ValueError, OverflowError):
            return None

    return number if math.isfinite(number) else None
