import math

import pytest

from kalden.core.datascience.validation import (
    _is_boolean,
    _is_numeric,
    parse_boolean,
    parse_finite_float,
)


@pytest.mark.parametrize(
    "value",
    [0, -4, 2.5, " 12.5 ", "0x10", b"-3", "1e-4"],
)
def test_is_numeric_accepts_supported_numeric_values(value) -> None:
    assert _is_numeric(value)


@pytest.mark.parametrize(
    "value",
    [True, False, None, "", "  ", "1.2.3", object()],
)
def test_is_numeric_rejects_non_numeric_values(value) -> None:
    assert not _is_numeric(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        (1, True),
        (0.0, False),
        (" TRUE ", True),
        ("false", False),
        ("1", True),
        ("0", False),
    ],
)
def test_boolean_validation_and_parsing(value, expected: bool) -> None:
    assert _is_boolean(value)
    assert parse_boolean(value) is expected


@pytest.mark.parametrize("value", [2, -1, "yes", "", None, math.nan])
def test_parse_boolean_rejects_unsupported_values(value) -> None:
    assert not _is_boolean(value)
    with pytest.raises(ValueError, match="Invalid boolean value"):
        parse_boolean(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (4, 4.0),
        (" -1.25 ", -1.25),
        ("0x10", 16.0),
        (b"2.5", 2.5),
    ],
)
def test_parse_finite_float_accepts_decimal_and_python_integer_syntax(
    value,
    expected: float,
) -> None:
    assert parse_finite_float(value) == pytest.approx(expected)


@pytest.mark.parametrize(
    "value",
    [True, False, "", "not-a-number", "nan", "inf", "-inf", object()],
)
def test_parse_finite_float_rejects_non_finite_or_invalid_values(value) -> None:
    assert parse_finite_float(value) is None
