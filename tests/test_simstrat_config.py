from io import StringIO
import json
import logging
from pathlib import Path
import re

import pandas as pd
import pytest

from kalden.core.simstrat import (
    SimstratConfig,
    SimstratConfigError,
    SimstratReadError,
    compute_sim_dates,
    configure_simstrat_logging,
    get_file_path_from_setup,
    get_simstrat_model_setups,
    read_simstrat_model_setup,
)


def _inflow_content(days: tuple[float, ...] = (0, 1, 2)) -> str:
    rows = "".join(
        f"\t{day:g}\t{day + 1:g}\t{day + 2:g}\t{day + 3:g}\n"
        for day in days
    )
    return (
        "\tTime [d]\tDeep one\tDeep two\tSurface\n"
        "\t2\t1\n"
        "-1\t\t-2\t-2\t0\n"
        f"{rows}"
    )


def _write_setup(
    tmp_path: Path,
    *,
    times: int | str = 1,
    depth_reference: str = "surface",
    morphology_datum: float | None = 500,
    couple_fabm: bool | str = False,
    legacy_fabm_path: bool = False,
) -> Path:
    root = tmp_path / "Lake"
    results = root / "Results"
    results.mkdir(parents=True)

    (root / "forcing.dat").write_text(
        "Time [d]\tAir temperature\n0\t10\n1\t11\n2\t12\n",
        encoding="utf-8",
    )
    (root / "morphology.dat").write_text(
        "Depth [m]\tArea [m2]\n0\t100\n-2\t80\n",
        encoding="utf-8",
    )
    (root / "initial.dat").write_text(
        "Depth [m]\tTemperature\n0\t20\n-2\t10\n",
        encoding="utf-8",
    )
    (root / "inflow.dat").write_text(_inflow_content(), encoding="utf-8")

    input_config: dict[str, str | float] = {
        "Forcing": "forcing.dat",
        "Morphology": "morphology.dat",
        "Initial conditions": "initial.dat",
        "Inflow": "inflow.dat",
    }
    if morphology_datum is not None:
        input_config["Morphology datum"] = morphology_datum

    config: dict[str, object] = {
        "Simulation": {
            "Reference year": 2024,
            "Start d": 0,
            "End d": 2,
            "Timestep s": 3600,
        },
        "ModelConfig": {
            "InflowMode": 1,
            "CoupleFABM": couple_fabm,
        },
        "Input": input_config,
        "Output": {
            "Path": "Results",
            "Times": times,
            "OutputDepthReference": depth_reference,
        },
    }

    if couple_fabm not in (False, "false", "0"):
        fabm_dir = root / "fabm"
        fabm_dir.mkdir()
        (fabm_dir / "oxygen.dat").write_text(
            _inflow_content(),
            encoding="utf-8",
        )
        path_key = "PathFABMinflow" if legacy_fabm_path else "FABMInflowPath"
        config["FABMConfig"] = {path_key: "fabm"}

    setup_path = root / "lake.par"
    setup_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return setup_path


def _write_output(
    setup_path: Path,
    *,
    name: str = "temperature.dat",
    times: tuple[float, ...] = (0, 1 / 48, 1 / 24),
    depths: tuple[int, ...] = (0, 1, 2),
) -> Path:
    path = setup_path.parent / "Results" / name
    header = ",".join(["Datetime", *(str(depth) for depth in depths)])
    rows = []
    for row_number, day in enumerate(times):
        values = [str(10 + row_number + offset) for offset in range(len(depths))]
        rows.append(",".join([f"{day:.15g}", *values]))
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


def test_setup_parsing_time_math_and_discovery(tmp_path: Path) -> None:
    setup_path = _write_setup(tmp_path)
    model = SimstratConfig(setup_path)

    assert read_simstrat_model_setup(setup_path)["Simulation"]["Reference year"] == 2024
    assert model.start_date == pd.Timestamp("2024-01-01")
    assert model.end_date == pd.Timestamp("2024-01-03")
    assert model.times[0] == pd.Timestamp("2024-01-01")
    assert model.times[-1] == pd.Timestamp("2024-01-03")
    assert len(model.times) == 49
    assert model.time_period_from_ts_number(3).total_seconds() == 10800

    assert get_simstrat_model_setups(tmp_path) == [str(setup_path.resolve())]
    assert get_simstrat_model_setups(tmp_path, exceptions=["lake"]) == []
    assert get_file_path_from_setup(setup_path, ("Output", "Path")) == str(
        setup_path.parent / "Results"
    )
    assert compute_sim_dates(
        "2024-01-01",
        "date_to_day_num",
        start_date="2024-01-02",
        end_date="2024-01-03 12:00",
    ) == (1, 2.5)
    assert compute_sim_dates(
        "2024-01-01",
        "day_num_to_date",
        day_num=1.5,
    ) == pd.Timestamp("2024-01-02 12:00")


def test_standard_logging_reports_reader_lifecycle(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    setup_path = _write_setup(tmp_path)
    _write_output(setup_path)

    with caplog.at_level(logging.DEBUG, logger="kalden.core.simstrat"):
        model = SimstratConfig(setup_path)
        model.load_inputs()
        model.load_outputs(sep=",")

    assert model.log.name == "kalden.core.simstrat.config.Lake"
    messages = [record.getMessage() for record in caplog.records]
    assert any("Reading Simstrat model setup" in message for message in messages)
    assert "Loading model setup inputs" in messages
    assert "Loaded 4 model input file(s)" in messages
    assert any("Loaded input Forcing" in message for message in messages)
    assert "Loaded 1 model output file(s)" in messages
    assert {record.levelno for record in caplog.records} >= {
        logging.INFO,
        logging.DEBUG,
    }


def test_configure_simstrat_logging_is_formatted_and_idempotent() -> None:
    stream = StringIO()
    package_logger = configure_simstrat_logging(stream=stream)

    try:
        configure_simstrat_logging(stream=stream)
        child_logger = logging.getLogger("kalden.core.simstrat.config.example")
        child_logger.info("formatted message")

        output = stream.getvalue()
        assert output.count("formatted message") == 1
        assert re.search(
            r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] "
            r"\[INFO\] kalden\.core\.simstrat\.config\.example: "
            r"formatted message",
            output,
        )
        console_handlers = [
            handler
            for handler in package_logger.handlers
            if handler.get_name() == "kalden.simstrat.console"
        ]
        assert len(console_handlers) == 1
    finally:
        for handler in list(package_logger.handlers):
            if handler.get_name() == "kalden.simstrat.console":
                package_logger.removeHandler(handler)
                handler.close()
        package_logger.setLevel(logging.NOTSET)
        package_logger.propagate = True


def test_setup_validation_rejects_invalid_schema_and_values(tmp_path: Path) -> None:
    setup_path = _write_setup(tmp_path)
    config = read_simstrat_model_setup(setup_path)
    config.pop("Input")
    setup_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(SimstratConfigError, match="section: Input"):
        SimstratConfig(setup_path)

    valid_model = SimstratConfig(_write_setup(tmp_path / "second"))
    with pytest.raises(ValueError, match="positive integer"):
        valid_model.time_period_from_ts_number(1.5)
    with pytest.raises(ValueError, match="operation must be"):
        compute_sim_dates("2024-01-01", "unsupported")


def test_output_time_file_supports_header_and_rejects_duplicates(
    tmp_path: Path,
) -> None:
    setup_path = _write_setup(tmp_path, times="output_times.dat")
    time_path = setup_path.parent / "output_times.dat"
    time_path.write_text("Time [d]\n0\n0.5\n2\n", encoding="utf-8")

    model = SimstratConfig(setup_path)
    assert model.times.tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-01 12:00"),
        pd.Timestamp("2024-01-03"),
    ]

    time_path.write_text("0\n0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate timestamps"):
        SimstratConfig(setup_path)


def test_inflow_reader_preserves_repeated_depths_and_exact_times(
    tmp_path: Path,
) -> None:
    path = tmp_path / "inflow.dat"
    path.write_text(_inflow_content(), encoding="utf-8")

    result = SimstratConfig.read_simstrat_inflow(
        path,
        ref_date="2024-01-01",
    )

    assert result.columns.tolist() == ["deep_-2", "deep_-2_2", "surface_0"]
    assert result.index.tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]
    assert result.iloc[-1].tolist() == [3, 4, 5]

    path.write_text(_inflow_content((1, 0)), encoding="utf-8")
    with pytest.raises(ValueError, match="not monotonic"):
        SimstratConfig.read_simstrat_inflow(path)


@pytest.mark.parametrize("legacy_fabm_path", [False, True])
def test_load_inputs_supports_physical_and_selma_files(
    tmp_path: Path,
    legacy_fabm_path: bool,
) -> None:
    setup_path = _write_setup(
        tmp_path,
        couple_fabm=True,
        legacy_fabm_path=legacy_fabm_path,
    )
    model = SimstratConfig(setup_path)

    inputs = model.load_inputs()

    assert set(inputs) == {
        "Forcing",
        "Morphology",
        "Initial conditions",
        "Inflow",
        "fabm_oxygen",
    }
    assert model.validate_inputs() == {}
    assert model.depths_bathy.tolist() == [0, -2]
    assert model.altitudes_bathy.tolist() == [500, 498]
    assert model.io_type("Forcing") == "input"
    assert inputs["Forcing"].index[-1] == pd.Timestamp("2024-01-03")


def test_input_loading_reports_missing_and_incomplete_files(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    setup_path = _write_setup(tmp_path)
    (setup_path.parent / "forcing.dat").unlink()

    with pytest.raises(SimstratReadError, match="Forcing"):
        SimstratConfig(setup_path).load_inputs()

    model = SimstratConfig(setup_path)
    with caplog.at_level(logging.WARNING, logger=model.log.name):
        with pytest.warns(RuntimeWarning, match="Forcing"):
            inputs = model.load_inputs(errors="warn")
    assert "Forcing" not in inputs
    assert any(
        record.levelno == logging.WARNING
        and "Could not read 'Forcing'" in record.getMessage()
        for record in caplog.records
    )

    (setup_path.parent / "forcing.dat").write_text(
        "Time [d]\tAir temperature\n0\t10\n1\t11\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ends too early"):
        SimstratConfig(setup_path).load_inputs()


def test_input_loading_rejects_duplicate_morphology_depths(tmp_path: Path) -> None:
    setup_path = _write_setup(tmp_path)
    (setup_path.parent / "morphology.dat").write_text(
        "Depth [m]\tArea [m2]\n0\t100\n0\t80\n",
        encoding="utf-8",
    )

    with pytest.raises(SimstratReadError, match="depths contain duplicates"):
        SimstratConfig(setup_path).load_inputs()


def test_load_outputs_preserves_subhour_times_and_builds_altitudes(
    tmp_path: Path,
) -> None:
    setup_path = _write_setup(tmp_path)
    _write_output(setup_path)
    _write_output(setup_path, name="salinity.dat")
    (setup_path.parent / "Results" / "_variables.dat").write_text(
        "Variable,Unit\ntemperature,C\n",
        encoding="utf-8",
    )
    (setup_path.parent / "Results" / "run.log").write_text(
        "not tabular",
        encoding="utf-8",
    )

    model = SimstratConfig(setup_path)
    outputs = model.load_outputs(sep=",")

    assert set(outputs) == {"salinity", "temperature"}
    assert outputs["temperature"].index.tolist() == [
        pd.Timestamp("2024-01-01 00:00"),
        pd.Timestamp("2024-01-01 00:30"),
        pd.Timestamp("2024-01-01 01:00"),
    ]
    assert outputs["temperature"].columns.tolist() == [0.0, -1.0, -2.0]
    assert model.time.equals(outputs["temperature"].index)
    assert model.depths_output.tolist() == [0, -1, -2]
    assert model.altitudes_output.tolist() == [500, 499, 498]
    assert model.depth_to_altitude is model.depth_to_altitude_table
    assert model.output_variables["Variable"].tolist() == ["temperature"]

    converted = model.depths_to_altitudes(outputs["temperature"])
    assert converted.columns.tolist() == [500, 499, 498]


def test_datetime_output_values_preserve_subsecond_precision(tmp_path: Path) -> None:
    setup_path = _write_setup(tmp_path)
    output_path = setup_path.parent / "Results" / "temperature.dat"
    output_path.write_text(
        "Datetime,0,1\n"
        "2024-01-01 00:00:00.123456789,10,9\n"
        "2024-01-01 00:30:00.987654321,11,10\n",
        encoding="utf-8",
    )

    result = SimstratConfig(setup_path).load_outputs(sep=",")["temperature"]

    assert result.index.tolist() == [
        pd.Timestamp("2024-01-01 00:00:00.123456789"),
        pd.Timestamp("2024-01-01 00:30:00.987654321"),
    ]


def test_bottom_referenced_outputs_are_shifted_to_surface_depths(
    tmp_path: Path,
) -> None:
    setup_path = _write_setup(tmp_path, depth_reference="bottom")
    _write_output(setup_path)

    result = SimstratConfig(setup_path).load_outputs(sep=",")["temperature"]

    assert result.columns.tolist() == [-2.0, -1.0, 0.0]


def test_output_resampling_uses_pandas3_safe_frequency_logic(
    tmp_path: Path,
) -> None:
    setup_path = _write_setup(tmp_path)
    _write_output(setup_path, times=(0, 1 / 24, 2 / 24))

    result = SimstratConfig(setup_path).load_outputs(
        sep=",",
        resample="30min",
    )["temperature"]

    assert result.index.tolist() == list(
        pd.date_range("2024-01-01", periods=5, freq="30min")
    )
    assert result.iloc[1, 0] == pytest.approx(10.5)


def test_output_loading_rejects_duplicate_timestamps(tmp_path: Path) -> None:
    setup_path = _write_setup(tmp_path)
    _write_output(setup_path, times=(0, 0))

    with pytest.raises(SimstratReadError, match="Duplicate timestamps"):
        SimstratConfig(setup_path).load_outputs(sep=",")


def test_altitude_mapping_is_optional(tmp_path: Path) -> None:
    setup_path = _write_setup(tmp_path, morphology_datum=None)
    _write_output(setup_path)
    model = SimstratConfig(setup_path)

    model.load_outputs(sep=",")

    assert model.depths_output.tolist() == [0, -1, -2]
    assert model.altitudes_output is None
    assert model.depth_to_altitude_table is None
    with pytest.raises(ValueError, match="No altitude mapping"):
        model.depths_to_altitudes(model.outputs["temperature"])


def test_reloading_empty_outputs_clears_previous_read_state(tmp_path: Path) -> None:
    setup_path = _write_setup(tmp_path)
    output_path = _write_output(setup_path)
    model = SimstratConfig(setup_path)
    model.load_outputs(sep=",")

    output_path.unlink()
    assert model.load_outputs(sep=",") == {}
    assert model.time is None
    assert model.depths_output is None
    assert model.altitudes_output is None
    assert model.depth_to_altitude_table is None


def test_invalid_fabm_flag_is_not_interpreted_by_string_truthiness(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    setup_path = _write_setup(tmp_path, couple_fabm="not-a-boolean")
    model = SimstratConfig(setup_path)

    with caplog.at_level(logging.ERROR, logger=model.log.name):
        with pytest.raises(SimstratConfigError, match="CoupleFABM"):
            model.load_inputs()

    assert any(
        record.levelno == logging.ERROR
        and "CoupleFABM must be a boolean" in record.getMessage()
        for record in caplog.records
    )
