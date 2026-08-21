from pathlib import Path

import pandas as pd
import pytest

from kalden.core.mike.res1d import (
    ObjectRef,
    _chainage_from_column,
    _default_logspace,
    _default_step_volume_m3,
    _normalize_load_mode,
    _normalize_object_type,
    _normalize_timeseries,
    _object_ref_from_source,
    _safe_cache_token,
    _select_reach_chainage,
    _split_special_reach_id,
    _storage_id_candidates,
    default_cache_dir,
    discharge_cutoff_sensitivity,
)


def test_default_cache_directory_is_next_to_result_file() -> None:
    result = Path("data") / "scenario.res1d"
    assert default_cache_dir(result) == Path("data") / "scenario_res1d_cache"


def test_load_mode_and_object_type_aliases_are_normalized() -> None:
    assert _normalize_load_mode(" LAZY ") == "filtered"
    assert _normalize_load_mode("full") == "full"
    assert _normalize_object_type("Branches") == "reach"
    assert _normalize_object_type("WEIRS") == "weir"

    with pytest.raises(ValueError, match="load_mode must be one of"):
        _normalize_load_mode("eager")
    with pytest.raises(ValueError, match="object_type must be one of"):
        _normalize_object_type("unknown")


def test_safe_cache_tokens_are_sanitized_and_collision_resistant() -> None:
    first = _safe_cache_token("Node/A:B")
    second = _safe_cache_token("Node A B")

    assert "/" not in first
    assert ":" not in first
    assert first != second
    assert _safe_cache_token("Node/A:B") == first


@pytest.mark.parametrize(
    ("raw_id", "expected"),
    [
        ("Weir:W1", ("weir", "W1", "Weir:W1")),
        (" pump : P-2 ", ("pump", "P-2", "Pump:P-2")),
        ("Valve: V3", ("valve", "V3", "Valve:V3")),
        ("ordinary reach", None),
    ],
)
def test_special_reach_ids_support_spacing_and_case(raw_id, expected) -> None:
    assert _split_special_reach_id(raw_id) == expected


def test_source_reach_is_exposed_as_special_object_reference() -> None:
    assert _object_ref_from_source("reach", "Pump: P1") == ObjectRef(
        object_type="pump",
        object_id="P1",
        source_object_type="reach",
        source_object_id="Pump: P1",
    )
    assert _object_ref_from_source("node", "N1") == ObjectRef(
        object_type="node",
        object_id="N1",
        source_object_type="node",
        source_object_id="N1",
    )


def test_storage_candidates_cover_compact_and_spaced_special_ids() -> None:
    assert _storage_id_candidates("weir", "W1") == [
        "W1",
        "Weir:W1",
        "Weir: W1",
    ]
    assert _storage_id_candidates("weir", "Weir:W1") == [
        "Weir:W1",
        "Weir: W1",
    ]


def test_normalize_timeseries_sorts_index_and_stringifies_columns() -> None:
    series = pd.Series(
        [2.0, 1.0],
        index=["2024-01-02", "2024-01-01"],
        name=7,
    )

    result = _normalize_timeseries(series)

    assert result.index.tolist() == pd.to_datetime(
        ["2024-01-01", "2024-01-02"]
    ).tolist()
    assert result.index.name == "time"
    assert result.columns.tolist() == ["7"]
    assert result.iloc[:, 0].tolist() == [1.0, 2.0]


def test_chainage_selection_supports_endpoints_and_center() -> None:
    frame = pd.DataFrame(
        {
            "Discharge:R1:10": [3.0, 5.0],
            "Discharge:R1:0": [1.0, 3.0],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )

    inlet = _select_reach_chainage(frame, "upstream", quantity="Discharge")
    outlet = _select_reach_chainage(frame, "downstream", quantity="Discharge")
    center = _select_reach_chainage(frame, "mean", quantity="Discharge")

    assert inlet.columns.tolist() == ["Discharge:inlet"]
    assert inlet.iloc[:, 0].tolist() == [1.0, 3.0]
    assert outlet.columns.tolist() == ["Discharge:outlet"]
    assert outlet.iloc[:, 0].tolist() == [3.0, 5.0]
    assert center.columns.tolist() == ["Discharge:mean"]
    assert center.iloc[:, 0].tolist() == [2.0, 4.0]
    assert _chainage_from_column("WaterLevel:R1:84.5") == 84.5
    assert _chainage_from_column("WaterLevel") is None


def test_chainage_selection_rejects_unknown_mode_or_unparseable_columns() -> None:
    frame = pd.DataFrame({"value": [1.0], "other": [2.0]})

    with pytest.raises(ValueError, match="chainage must be one of"):
        _select_reach_chainage(frame, "quarter", quantity="Q")
    with pytest.raises(ValueError, match="Could not infer chainages"):
        _select_reach_chainage(frame, "inlet", quantity="Q")


def test_default_step_volume_integrates_sorted_discharge() -> None:
    discharge = pd.Series(
        [3.0, 1.0, 2.0],
        index=pd.to_datetime(
            ["2024-01-01 02:00", "2024-01-01 00:00", "2024-01-01 01:00"]
        ),
    )

    assert _default_step_volume_m3(discharge) == pytest.approx(10_800)
    assert _default_step_volume_m3(discharge.iloc[:0]) == 0.0


def test_default_logspace_includes_endpoints_and_validates_bounds() -> None:
    values = _default_logspace(1e-3, 1e1, 5)

    assert values[0] == pytest.approx(1e-3)
    assert values[-1] == pytest.approx(1e1)
    assert len(values) == 5
    assert _default_logspace(2, 4, 1) == [2.0]

    with pytest.raises(ValueError, match="strictly positive"):
        _default_logspace(0, 1, 5)


def test_discharge_cutoff_sensitivity_detects_material_volume_change() -> None:
    discharge = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.date_range("2024-01-01", periods=3, freq="h"),
    )

    sensitivity, first_break, threshold = discharge_cutoff_sensitivity(
        discharge,
        cutoffs=[0.0, 0.5, 1.5],
        break_threshold=0.1,
    )

    assert sensitivity["volume_m3"].tolist() == [10_800.0, 10_800.0, 7_200.0]
    assert threshold == pytest.approx(0.1)
    assert first_break is not None
    assert first_break["cutoff_m3s"] == pytest.approx(1.5)
