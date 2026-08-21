from kalden import hello
from kalden.core.datascience.validation import _is_numeric, parse_finite_float
from kalden.core.datascience.generic import is_numeric


def test_hello() -> None:
    assert hello("World") == "Hello, World!"


def test_boolean_is_not_numeric() -> None:
    assert not _is_numeric(True)
    assert not _is_numeric(False)
    assert parse_finite_float(True) is None
    assert parse_finite_float(False) is None
    assert not is_numeric(True)
    assert not is_numeric(False)
