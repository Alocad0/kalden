from kalden import hello
from kalden.core.datascience.validation import _is_numeric, parse_finite_float


def test_hello() -> None:
    assert hello("World") == "Hello, World!"


def test_boolean_is_not_numeric() -> None:
    assert not _is_numeric(True)
    assert not _is_numeric(False)
    assert parse_finite_float(True) is None
    assert parse_finite_float(False) is None
