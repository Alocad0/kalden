from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kalden.core.simstrat import (
    SimstratConfig,
    generate_inflow_content,
    inputs_generate_content,
    mgL_to_mmolm3,
    mg_l_to_mmol_m3,
    sum_complete_flows,
    write_inflow_file,
)


def _flow_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "small": [0.00456789123, 0.00567891234],
            "large": [12.34567891, 23.45678912],
        },
        index=pd.date_range("2024-01-01", periods=2, freq="h"),
    )


def _flow_definitions() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    deep = [{"depth": -2, "col": "small", "header": "Small [mmol/m3]"}]
    surface = [{"depth": 0, "col": "large", "header": "Large [m3/s]"}]
    return deep, surface


def test_inflow_content_preserves_small_values_and_round_trips(
    tmp_path: Path,
) -> None:
    frame = _flow_frame()
    deep, surface = _flow_definitions()

    content = generate_inflow_content(
        frame,
        "2024-01-01",
        deep,
        surface,
    )
    path = tmp_path / "Qin.dat"
    path.write_text(content, encoding="utf-8")
    loaded = SimstratConfig.read_simstrat_inflow(
        path,
        ref_date="2024-01-01",
    )

    assert "0.00456789123" in content
    assert "12.34567891" in content
    assert loaded.iloc[0, 0] == pytest.approx(frame.iloc[0, 0])
    assert loaded.iloc[1, 1] == pytest.approx(frame.iloc[1, 1])


def test_legacy_writer_alias_uses_precision_safe_default() -> None:
    frame = _flow_frame()
    deep, surface = _flow_definitions()

    content = inputs_generate_content(
        frame,
        pd.Timestamp("2024-01-01"),
        deep,
        surface,
    )

    assert "0.00456789123" in content
    assert "\t0.00\t" not in content


def test_live_writer_replaces_atomically_and_keeps_original_on_validation_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "Qin.dat"
    path.write_text("original", encoding="utf-8")
    frame = _flow_frame()
    deep, surface = _flow_definitions()

    result = write_inflow_file(
        path,
        frame,
        "2024-01-01",
        deep,
        surface,
    )

    assert result == path.resolve()
    assert "0.00456789123" in path.read_text(encoding="utf-8")
    assert list(tmp_path.glob(".*.tmp")) == []

    original = path.read_text(encoding="utf-8")
    invalid = frame.copy()
    invalid.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="Missing value"):
        write_inflow_file(
            path,
            invalid,
            "2024-01-01",
            deep,
            surface,
        )
    assert path.read_text(encoding="utf-8") == original


def test_writer_can_encode_explicit_missing_values_when_requested(
    tmp_path: Path,
) -> None:
    frame = _flow_frame()
    frame.iloc[0, 0] = np.nan
    deep, surface = _flow_definitions()
    path = tmp_path / "Qin.dat"

    write_inflow_file(
        path,
        frame,
        "2024-01-01",
        deep,
        surface,
        allow_missing=True,
    )
    loaded = SimstratConfig.read_simstrat_inflow(
        path,
        ref_date="2024-01-01",
    )

    assert pd.isna(loaded.iloc[0, 0])


def test_complete_flow_sum_rejects_partial_totals() -> None:
    frame = pd.DataFrame(
        {"river": [1.0, 2.0], "tributary": [3.0, np.nan]},
        index=pd.date_range("2024-01-01", periods=2, freq="h"),
    )

    with pytest.raises(ValueError, match="missing flow components"):
        sum_complete_flows(frame)

    result = sum_complete_flows(frame, missing="keep")
    assert result.iloc[0] == 4.0
    assert pd.isna(result.iloc[1])


def test_oxygen_conversion_does_not_round_to_whole_units() -> None:
    result = mg_l_to_mmol_m3(0.155, 32.0)

    assert result == pytest.approx(4.84375)
    assert result != round(result)
    assert mgL_to_mmolm3(0.155, 32.0) == pytest.approx(result)


def test_writer_rejects_ambiguous_duplicate_columns() -> None:
    frame = _flow_frame()
    frame.columns = ["flow", "flow"]
    definitions = [{"depth": -2, "col": "flow", "header": "Q"}]

    with pytest.raises(ValueError, match="duplicate labels"):
        generate_inflow_content(
            frame,
            "2024-01-01",
            definitions,
            [],
        )
