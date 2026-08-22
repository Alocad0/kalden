"""Core readers for Simstrat and Simstrat-SELMA model setups.

The setup file uses JSON despite its conventional ``.par`` extension. This
module focuses on deterministic configuration and tabular I/O. Notebook UI,
plotting, derived lake diagnostics, and serialization belong in higher-level
modules and are intentionally not imported here.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
import json
import logging
import math
from pathlib import Path
import re
from typing import Any, Literal, TextIO
import warnings

import numpy as np
import pandas as pd

from kalden.core.datascience.validation import parse_boolean, parse_finite_float
from kalden.core.io import detect_file_encoding, file_has_content

PathLike = str | Path
ErrorMode = Literal["raise", "warn", "ignore"]
SIMSTRAT_LOGGER_NAME = "kalden.core.simstrat"
_CONSOLE_HANDLER_NAME = "kalden.simstrat.console"

_library_logger = logging.getLogger(SIMSTRAT_LOGGER_NAME)
if not any(
    isinstance(handler, logging.NullHandler)
    for handler in _library_logger.handlers
):
    _library_logger.addHandler(logging.NullHandler())

logger = logging.getLogger(__name__)

__all__ = [
    "SimstratConfig",
    "SimstratConfigError",
    "SimstratReadError",
    "compute_sim_dates",
    "configure_simstrat_logging",
    "get_file_path_from_setup",
    "get_simstrat_model_setups",
    "read_simstrat_model_setup",
]


def configure_simstrat_logging(
    level: int | str = logging.INFO,
    *,
    stream: TextIO | None = None,
    propagate: bool = False,
) -> logging.Logger:
    """Enable timestamped console logging for the Simstrat logger hierarchy.

    Applications with an existing logging configuration do not need this
    helper. It is primarily a convenience for scripts and notebooks and is
    idempotent: repeated calls update the same console handler.
    """
    package_logger = logging.getLogger(SIMSTRAT_LOGGER_NAME)
    handler = next(
        (
            candidate
            for candidate in package_logger.handlers
            if candidate.get_name() == _CONSOLE_HANDLER_NAME
        ),
        None,
    )
    if handler is None:
        handler = logging.StreamHandler(stream)
        handler.set_name(_CONSOLE_HANDLER_NAME)
        package_logger.addHandler(handler)
    elif stream is not None and isinstance(handler, logging.StreamHandler):
        handler.setStream(stream)

    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    package_logger.setLevel(level)
    package_logger.propagate = propagate
    return package_logger


class SimstratConfigError(ValueError):
    """Raised when a Simstrat setup is missing or internally inconsistent."""


class SimstratReadError(RuntimeError):
    """Raised when a referenced Simstrat input or output cannot be read."""


def _validate_error_mode(errors: ErrorMode) -> ErrorMode:
    if errors not in {"raise", "warn", "ignore"}:
        raise ValueError("errors must be one of: 'raise', 'warn', 'ignore'.")
    return errors


def _normalized_encoding(path: Path) -> str:
    return detect_file_encoding(path) or "utf-8"


def _day_number(value: float) -> int | float:
    return int(value) if float(value).is_integer() else float(value)


def _timedelta_from_decimal_days(
    values: float | Sequence[float] | pd.Series,
) -> pd.Timedelta | pd.TimedeltaIndex:
    """Convert decimal days without retaining binary-float nanosecond noise."""
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("Time values must all be finite numbers.")
    microseconds = np.rint(array * 86_400_000_000).astype(np.int64)
    converted = pd.to_timedelta(microseconds, unit="us")
    if array.ndim == 0:
        return converted
    return converted


class SimstratConfig:
    """Read one Simstrat or Simstrat-SELMA model setup.

    Construction parses only the JSON setup and its simulation time axis. Call
    :meth:`load_inputs` and :meth:`load_outputs` explicitly for model data.

    Parameters
    ----------
    par_file:
        Path to the JSON-formatted Simstrat ``.par`` setup.
    notebook_mode:
        Accepted for compatibility with the former notebook utility. Standard
        logging is controlled by the application or
        :func:`configure_simstrat_logging`, not by this flag.
    """

    input_file_exceptions: set[str] = {"SetFABMDiagnosticVars.dat"}
    output_file_exceptions: set[str] = {
        "fabm_list_diagnostic_horizontal.dat",
        "fabm_list_diagnostic_interior.dat",
    }
    output_suffix_exceptions: set[str] = {
        ".json",
        ".log",
        ".nc",
        ".yaml",
        ".yml",
    }

    def __init__(self, par_file: PathLike, notebook_mode: bool = False) -> None:
        del notebook_mode

        par_path = Path(par_file).expanduser().resolve(strict=True)
        if not par_path.is_file():
            raise SimstratConfigError(f"Not a regular setup file: {par_path}")
        if par_path.suffix.lower() != ".par":
            raise SimstratConfigError(
                f"Expected a Simstrat .par setup, got: {par_path}"
            )

        self.par_path = par_path
        self.par_file = str(par_path)
        self.root_path = par_path.parent
        self.root = str(self.root_path)
        self.name = self.root_path.name
        self.log = logging.getLogger(f"{__name__}.{self.name}")
        self.log.info("Reading Simstrat model setup from %s", self.par_file)

        self.config: dict[str, Any] = {}
        self.inputs: dict[str, pd.DataFrame | str] = {}
        self.inputs_paths: dict[str, str] = {}
        self.outputs: dict[str, pd.DataFrame] = {}
        self.output_variables: pd.DataFrame | None = None
        self.outputs_loaded = False
        self.time: pd.DatetimeIndex | None = None

        self.depths_bathy: np.ndarray | None = None
        self.altitudes_bathy: np.ndarray | None = None
        self.depths_output: np.ndarray | None = None
        self.altitudes_output: np.ndarray | None = None
        self.depth_to_altitude_table: pd.Series | None = None

        self.simulation_log: str | None = None
        self.execution_time: datetime | None = None

        try:
            self.parse()
            self.get_simulation_times()
            self.result_path = self.resolve_config_path(("Output", "Path"))
            self.result_dir = str(self.result_path)
            self.load_log(required=False)
            self.get_execution_date()
        except Exception as exc:
            self.log.error("Could not initialize model setup: %s", exc)
            raise

        self.log.info(
            "Initialized model setup for %s to %s",
            self.start_date,
            self.end_date,
        )

    @staticmethod
    def read_simstrat_model_setup(config_path: PathLike) -> dict[str, Any]:
        """Read a JSON-formatted Simstrat ``.par`` file."""
        path = Path(config_path).expanduser().resolve(strict=True)
        if not path.is_file():
            raise SimstratConfigError(f"Not a regular setup file: {path}")

        try:
            with path.open("r", encoding="utf-8-sig") as handle:
                config = json.load(handle)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SimstratConfigError(
                f"Could not read Simstrat setup {path}: {exc}"
            ) from exc

        if not isinstance(config, dict):
            raise SimstratConfigError(
                f"Simstrat setup must contain a JSON object: {path}"
            )
        logger.debug("Read Simstrat setup file %s", path)
        return config

    @staticmethod
    def get_nested_value(data: Mapping[str, Any], keys: Sequence[str]) -> Any:
        """Return a nested mapping value with a useful missing-key message."""
        value: Any = data
        traversed: list[str] = []
        for key in keys:
            traversed.append(str(key))
            if not isinstance(value, Mapping) or key not in value:
                joined = "/".join(traversed)
                raise SimstratConfigError(
                    f"Missing configuration key: {joined}"
                )
            value = value[key]
        return value

    @classmethod
    def get_simstrat_model_setups(
        cls,
        root_dir: PathLike,
        exceptions: Sequence[str] | None = None,
    ) -> list[str]:
        """Find ``.par`` files recursively in deterministic order."""
        root = Path(root_dir).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise NotADirectoryError(f"Not a directory: {root}")

        excluded = tuple(item.casefold() for item in (exceptions or ()))
        paths = [
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.casefold() == ".par"
            and not any(token in str(path).casefold() for token in excluded)
        ]
        result = [str(path) for path in sorted(paths, key=lambda p: str(p).casefold())]
        logger.info("Found %d Simstrat setup file(s) below %s", len(result), root)
        return result

    @classmethod
    def get_file_path_from_setup(
        cls,
        setup_path: PathLike,
        keys: Sequence[str],
    ) -> str:
        """Resolve a path-valued setting relative to its setup file."""
        path = Path(setup_path).expanduser().resolve(strict=True)
        setup = cls.read_simstrat_model_setup(path)
        value = cls.get_nested_value(setup, keys)
        return str(cls._resolve_path_value(path.parent, value))

    @staticmethod
    def _resolve_path_value(root: Path, value: Any) -> Path:
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise SimstratConfigError(
                f"Expected a non-empty file path, got {value!r}."
            )
        path = Path(value).expanduser()
        return path if path.is_absolute() else root / path

    def resolve_config_path(
        self,
        keys: Sequence[str],
        *,
        must_exist: bool = False,
    ) -> Path:
        """Resolve a path-valued setting relative to this setup."""
        value = self.get_nested_value(self.config, keys)
        path = self._resolve_path_value(self.root_path, value)
        return path.resolve(strict=must_exist)

    def get_file_path(self, keys: Sequence[str]) -> str:
        """Compatibility wrapper returning a resolved string path."""
        return str(self.resolve_config_path(keys))

    def parse(self) -> None:
        """Parse and validate core configuration fields."""
        self.config = self.read_simstrat_model_setup(self.par_path)

        for section in ("Simulation", "Input", "Output"):
            value = self.config.get(section)
            if not isinstance(value, Mapping):
                raise SimstratConfigError(
                    f"Missing or invalid configuration section: {section}"
                )

        simulation = self.config["Simulation"]
        required = ("Reference year", "Start d", "End d", "Timestep s")
        parsed: dict[str, float] = {}
        for key in required:
            number = parse_finite_float(simulation.get(key))
            if number is None:
                raise SimstratConfigError(
                    f"Simulation/{key} must be a finite number."
                )
            parsed[key] = number

        reference_year = parsed["Reference year"]
        if not reference_year.is_integer():
            raise SimstratConfigError("Simulation/Reference year must be an integer.")
        if parsed["End d"] < parsed["Start d"]:
            raise SimstratConfigError(
                "Simulation/End d must be greater than or equal to Start d."
            )
        if parsed["Timestep s"] <= 0:
            raise SimstratConfigError("Simulation/Timestep s must be positive.")

        self.reference_year = int(reference_year)
        self.start_day = _day_number(parsed["Start d"])
        self.end_day = _day_number(parsed["End d"])
        self.timestep_seconds = float(parsed["Timestep s"])

        model_config = self.config.get("ModelConfig", {})
        self.inflow_mode = (
            model_config.get("InflowMode")
            if isinstance(model_config, Mapping)
            else None
        )

        morphology_datum = self.config["Input"].get("Morphology datum")
        self.morphology_datum = parse_finite_float(morphology_datum)
        if morphology_datum is not None and self.morphology_datum is None:
            raise SimstratConfigError(
                "Input/Morphology datum must be a finite number when provided."
            )
        self.log.debug(
            "Parsed setup: reference year=%d, timestep=%g s, inflow mode=%r",
            self.reference_year,
            self.timestep_seconds,
            self.inflow_mode,
        )

    def get_simulation_times(self) -> pd.DatetimeIndex | None:
        """Compute or read the configured output time axis."""
        self.reference_date = datetime(self.reference_year, 1, 1)
        self.start_date = self.reference_date + timedelta(days=float(self.start_day))
        self.end_date = self.reference_date + timedelta(days=float(self.end_day))

        output_times = self.config["Output"].get("Times")
        numeric_times = parse_finite_float(output_times)

        if numeric_times is not None:
            if numeric_times <= 0:
                raise SimstratConfigError("Output/Times must be positive.")
            frequency = pd.Timedelta(
                seconds=self.timestep_seconds * numeric_times
            )
            self.times = pd.date_range(
                start=self.start_date,
                end=self.end_date,
                freq=frequency,
            )
            self.log.info(
                "Configured %d output timestamp(s) at %s intervals",
                len(self.times),
                frequency,
            )
            return self.times

        if output_times is None or not str(output_times).strip():
            self.times = None
            self.log.warning("No configured output time axis was found")
            return None

        path = self._resolve_path_value(self.root_path, output_times).resolve(
            strict=True
        )
        self.times = self._read_time_axis(path)
        self.log.info(
            "Read %d configured output timestamp(s) from %s",
            len(self.times),
            path,
        )
        return self.times

    def _read_time_axis(self, path: Path) -> pd.DatetimeIndex:
        encoding = _normalized_encoding(path)
        values: list[pd.Timestamp] = []

        with path.open("r", encoding=encoding) as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.split("#", 1)[0].strip()
                if not line:
                    continue
                token = re.split(r"[;,\s]+", line, maxsplit=1)[0]
                numeric = parse_finite_float(token)
                if numeric is not None:
                    values.append(
                        pd.Timestamp(self.reference_date)
                        + _timedelta_from_decimal_days(numeric)
                    )
                    continue
                try:
                    values.append(pd.Timestamp(token))
                except (TypeError, ValueError) as exc:
                    if not values and token.casefold() in {
                        "date",
                        "datetime",
                        "time",
                        "times",
                    }:
                        continue
                    raise SimstratConfigError(
                        f"Invalid output time at {path}:{line_number}: {token!r}"
                    ) from exc

        if not values:
            raise SimstratConfigError(f"Output time file is empty: {path}")

        index = pd.DatetimeIndex(values, name="time")
        self._validate_time_index(index, source=path)
        return index

    @staticmethod
    def compute_sim_dates(
        ref_date: str | datetime | pd.Timestamp,
        operation: str,
        start_date: str | datetime | pd.Timestamp | None = None,
        end_date: str | datetime | pd.Timestamp | None = None,
        day_num: float | None = None,
    ) -> tuple[int | float, int | float] | pd.Timestamp:
        """Convert between datetimes and days relative to a reference date."""
        reference = pd.Timestamp(ref_date)

        if operation == "date_to_day_num":
            if start_date is None or end_date is None:
                raise ValueError(
                    "start_date and end_date are required for date_to_day_num."
                )
            start = (pd.Timestamp(start_date) - reference).total_seconds() / 86400
            end = (pd.Timestamp(end_date) - reference).total_seconds() / 86400
            return _day_number(start), _day_number(end)

        if operation == "day_num_to_date":
            if day_num is None or not math.isfinite(float(day_num)):
                raise ValueError(
                    "A finite day_num is required for day_num_to_date."
                )
            return reference + pd.to_timedelta(day_num, unit="D")

        raise ValueError(
            "operation must be 'date_to_day_num' or 'day_num_to_date'."
        )

    def time_period_from_ts_number(self, num_timesteps: int) -> timedelta:
        """Convert a positive number of model timesteps to a duration."""
        parsed = parse_finite_float(num_timesteps)
        if parsed is None or parsed <= 0 or not parsed.is_integer():
            raise ValueError("num_timesteps must be a positive integer.")
        return timedelta(seconds=self.timestep_seconds * int(parsed))

    @staticmethod
    def _split_fields(line: str, sep: str) -> list[str]:
        if not sep:
            raise ValueError("sep must be a non-empty string.")
        return line.rstrip("\r\n").split(sep)

    @staticmethod
    def _rstrip_empty_fields(fields: Sequence[str]) -> list[str]:
        result = list(fields)
        while result and result[-1] == "":
            result.pop()
        return result

    @staticmethod
    def _make_flow_names(flow_type: str, depths: Sequence[str]) -> list[str]:
        seen: defaultdict[str, int] = defaultdict(int)
        names: list[str] = []
        for depth in depths:
            seen[depth] += 1
            suffix = f"_{seen[depth]}" if seen[depth] > 1 else ""
            names.append(f"{flow_type}_{depth}{suffix}")
        return names

    @classmethod
    def _read_inflow_header(
        cls,
        lines: Sequence[str],
        *,
        sep: str,
        filename: str,
    ) -> tuple[int, int, list[str]]:
        line_two = cls._rstrip_empty_fields(cls._split_fields(lines[1], sep))
        if line_two and line_two[0] == "":
            line_two = line_two[1:]
        if len(line_two) < 2:
            raise ValueError(
                f"{filename}: line 2 must contain n_deep and n_surface."
            )

        try:
            n_deep = int(float(line_two[0]))
            n_surface = int(float(line_two[1]))
        except ValueError as exc:
            raise ValueError(
                f"{filename}: could not parse n_deep and n_surface on line 2."
            ) from exc
        if n_deep < 0 or n_surface < 0:
            raise ValueError(f"{filename}: flow counts cannot be negative.")

        line_three = cls._rstrip_empty_fields(cls._split_fields(lines[2], sep))
        if not line_three or line_three[0].strip() != "-1":
            raise ValueError(f"{filename}: line 3 must start with -1.")

        depth_tokens = line_three[1:]
        if depth_tokens and depth_tokens[0] == "":
            depth_tokens = depth_tokens[1:]
        depth_tokens = [token.strip() for token in depth_tokens]

        expected = n_deep + n_surface
        if len(depth_tokens) != expected or any(not token for token in depth_tokens):
            raise ValueError(
                f"{filename}: expected {expected} non-empty depths on line 3, "
                f"found {len(depth_tokens)}."
            )

        names = cls._make_flow_names("deep", depth_tokens[:n_deep])
        names.extend(cls._make_flow_names("surface", depth_tokens[n_deep:]))
        return n_deep, n_surface, names

    @classmethod
    def read_simstrat_inflow(
        cls,
        path: PathLike,
        ref_date: str | datetime | pd.Timestamp | None = None,
        set_index: bool = True,
        check_monotonic: bool = True,
        sep: str = "\t",
        ignore_empty_cols: bool = False,
    ) -> pd.DataFrame:
        """Read a Simstrat multi-depth inflow or outflow table."""
        source = Path(path).expanduser().resolve(strict=True)
        encoding = _normalized_encoding(source)
        with source.open("r", encoding=encoding, errors="strict") as handle:
            lines = [line.rstrip("\r\n") for line in handle if line.strip()]

        if len(lines) < 4:
            raise ValueError(
                f"{source}: file is too short for a Simstrat inflow table."
            )

        n_deep, n_surface, data_columns = cls._read_inflow_header(
            lines,
            sep=sep,
            filename=source.name,
        )
        expected = 1 + n_deep + n_surface
        rows: list[list[float]] = []

        for line_number, line in enumerate(lines[3:], start=4):
            fields = cls._split_fields(line, sep)
            if fields and fields[0] == "":
                fields = fields[1:]

            while fields and fields[-1] == "" and len(fields) > expected:
                if not ignore_empty_cols:
                    break
                fields.pop()

            if len(fields) < expected:
                fields.extend([""] * (expected - len(fields)))
            if len(fields) != expected:
                raise ValueError(
                    f"{source}: line {line_number} has {len(fields)} fields; "
                    f"expected {expected}."
                )

            row: list[float] = []
            for token in fields:
                stripped = token.strip()
                if not stripped:
                    row.append(np.nan)
                    continue
                number = parse_finite_float(stripped)
                if number is None:
                    raise ValueError(
                        f"{source}: invalid numeric value {token!r} "
                        f"on line {line_number}."
                    )
                row.append(number)
            rows.append(row)

        frame = pd.DataFrame(rows, columns=["days", *data_columns])
        if frame.empty:
            raise ValueError(f"{source}: inflow table contains no data rows.")
        if check_monotonic:
            if not frame["days"].is_monotonic_increasing:
                raise ValueError(f"{source}: time values are not monotonic.")
            if frame["days"].duplicated().any():
                raise ValueError(f"{source}: time values contain duplicates.")

        if ref_date is not None:
            index = pd.Timestamp(ref_date) + _timedelta_from_decimal_days(
                frame["days"]
            )
            frame.insert(0, "datetime", index)
            if set_index:
                frame = frame.set_index("datetime").drop(columns="days")
                frame.index.name = "time"
        elif set_index:
            frame = frame.set_index("days")

        return frame

    @staticmethod
    def _strip_column_names(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        result.columns = [
            column.strip() if isinstance(column, str) else column
            for column in result.columns
        ]
        return result

    @staticmethod
    def _resolve_column(frame: pd.DataFrame, requested: str) -> Any:
        matches = [
            column
            for column in frame.columns
            if str(column).strip().casefold() == requested.casefold()
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one {requested!r} column, found {matches}. "
                f"Available columns: {list(frame.columns)}"
            )
        return matches[0]

    @staticmethod
    def _looks_like_inflow(path: Path, sep: str) -> bool:
        encoding = _normalized_encoding(path)
        with path.open("r", encoding=encoding) as handle:
            first_line = handle.readline()
        return first_line.startswith(sep)

    def _read_regular_input(
        self,
        key: str,
        path: Path,
        *,
        sep: str,
        date_col: str,
    ) -> pd.DataFrame:
        encoding = _normalized_encoding(path)
        frame = pd.read_csv(path, sep=sep, encoding=encoding)
        frame = self._strip_column_names(frame)

        if key == "Forcing":
            time_column = self._resolve_column(frame, date_col)
            days = pd.to_numeric(frame[time_column], errors="raise")
            index = pd.Timestamp(self.reference_date) + _timedelta_from_decimal_days(
                days
            )
            frame = frame.drop(columns=time_column)
            frame.insert(0, "days", days.to_numpy())
            frame.index = pd.DatetimeIndex(index, name="time")
            self._validate_time_index(frame.index, source=path)

        if key == "Initial conditions":
            try:
                depth_column = self._resolve_column(frame, "Depth [m]")
            except ValueError:
                depth_column = frame.columns[0]
            frame = frame.set_index(depth_column)

        return frame

    def _record_morphology(self, frame: pd.DataFrame) -> None:
        depth_column = self._resolve_column(frame, "Depth [m]")
        depths = pd.to_numeric(frame[depth_column], errors="raise").to_numpy(
            dtype=float,
            copy=True,
        )
        depths[depths == 0] = 0
        self.depths_bathy = depths
        if pd.Index(depths).has_duplicates:
            raise ValueError("Morphology depths contain duplicates.")
        if self.morphology_datum is not None:
            self.altitudes_bathy = depths + self.morphology_datum

    def _warn(self, message: str, *, stacklevel: int = 2) -> None:
        """Emit one logging record while preserving the warnings API."""
        self.log.warning("%s", message)
        warnings.warn(
            message,
            RuntimeWarning,
            stacklevel=stacklevel + 1,
        )

    def _handle_read_error(
        self,
        name: str,
        path: Path,
        exc: Exception,
        errors: ErrorMode,
    ) -> None:
        message = f"Could not read {name!r} from {path}: {exc}"
        if errors == "raise":
            self.log.error("%s", message)
            raise SimstratReadError(message) from exc
        if errors == "warn":
            self._warn(message, stacklevel=3)
        else:
            self.log.debug("Ignored read failure: %s", message)

    def load_inputs(
        self,
        sep: str = "\t",
        date_col: str = "Time [d]",
        *,
        errors: ErrorMode = "raise",
        validate: bool = True,
    ) -> dict[str, pd.DataFrame | str]:
        """Load physical and optional SELMA/FABM inputs."""
        errors = _validate_error_mode(errors)
        self.log.info("Loading model setup inputs")
        self.inputs = {}
        self.inputs_paths = {}
        self.depths_bathy = None
        self.altitudes_bathy = None

        candidates: list[tuple[str, Path]] = []
        for key, value in self.config["Input"].items():
            if parse_finite_float(value) is not None or not isinstance(value, str):
                continue
            path = self._resolve_path_value(self.root_path, value)
            if path.is_dir() or path.name in self.input_file_exceptions:
                continue
            candidates.append((str(key), path))

        ignored_input_names = sorted(self.input_file_exceptions)
        if ignored_input_names:
            self.log.info("Ignoring configured input files: %s", ignored_input_names)

        model_config = self.config.get("ModelConfig", {})
        couple_fabm_value = (
            model_config.get("CoupleFABM", False)
            if isinstance(model_config, Mapping)
            else False
        )
        try:
            couple_fabm = parse_boolean(couple_fabm_value)
        except ValueError as exc:
            message = "ModelConfig/CoupleFABM must be a boolean or 0/1."
            self.log.error("%s", message)
            raise SimstratConfigError(message) from exc
        if couple_fabm:
            self.log.info("Loading coupled FABM/SELMA inputs")
            fabm_config = self.config.get("FABMConfig")
            if not isinstance(fabm_config, Mapping):
                message = (
                    "ModelConfig/CoupleFABM is enabled but FABMConfig is missing."
                )
                self.log.error("%s", message)
                raise SimstratConfigError(message)
            inflow_dir_value = fabm_config.get("FABMInflowPath")
            if inflow_dir_value is None:
                inflow_dir_value = fabm_config.get("PathFABMinflow")
            if inflow_dir_value is None:
                message = (
                    "CoupleFABM is enabled but FABMConfig/FABMInflowPath "
                    "is missing."
                )
                self.log.error("%s", message)
                raise SimstratConfigError(message)
            inflow_dir = self._resolve_path_value(
                self.root_path,
                inflow_dir_value,
            )
            if not inflow_dir.is_dir():
                message = f"FABM inflow directory does not exist: {inflow_dir}"
                self.log.error("%s", message)
                raise SimstratConfigError(message)
            candidates.extend(
                (f"fabm_{path.stem}", path)
                for path in sorted(inflow_dir.iterdir(), key=lambda p: p.name.casefold())
                if path.is_file() and path.name not in self.input_file_exceptions
            )

        for key, path in candidates:
            try:
                if not path.is_file():
                    raise FileNotFoundError(path)
                if key.startswith("fabm_") or self._looks_like_inflow(path, sep):
                    frame = self.read_simstrat_inflow(
                        path,
                        ref_date=self.reference_date,
                        sep=sep,
                        ignore_empty_cols=True,
                    )
                else:
                    frame = self._read_regular_input(
                        key,
                        path,
                        sep=sep,
                        date_col=date_col,
                    )

                self.inputs[key] = frame
                self.inputs_paths[key] = str(path.resolve())
                if key == "Morphology":
                    self._record_morphology(frame)
                self.log.debug(
                    "Loaded input %s from %s (%d row(s), %d column(s))",
                    key,
                    path,
                    frame.shape[0],
                    frame.shape[1],
                )
            except Exception as exc:
                self._handle_read_error(key, path, exc, errors)

        if validate:
            problems = self.validate_inputs(raise_error=False)
            if problems:
                if errors == "raise":
                    self.validate_inputs(raise_error=True)
                if errors == "warn":
                    self._warn(
                        "Some Simstrat inputs do not cover the simulation period: "
                        + "; ".join(
                            f"{key}: {value}" for key, value in problems.items()
                        ),
                        stacklevel=2,
                    )
                if errors == "ignore":
                    self.log.debug("Ignored input validation problems: %s", problems)
        self.log.info("Loaded %d model input file(s)", len(self.inputs))
        return self.inputs

    def validate_inputs(
        self,
        inputs: Mapping[str, pd.DataFrame | str] | None = None,
        *,
        raise_error: bool = True,
    ) -> dict[str, str]:
        """Validate timestamp order, uniqueness, and simulation coverage."""
        self.log.info("Validating model inputs")
        selected = self.inputs if inputs is None else inputs
        problems: dict[str, str] = {}

        for name, value in selected.items():
            if not isinstance(value, pd.DataFrame):
                continue
            if value.empty:
                problems[name] = "empty dataframe"
                continue
            if not isinstance(value.index, pd.DatetimeIndex):
                continue

            issues: list[str] = []
            if not value.index.is_monotonic_increasing:
                issues.append("timestamps are not sorted")
            if value.index.has_duplicates:
                issues.append("timestamps contain duplicates")
            if value.index.min() > pd.Timestamp(self.start_date):
                issues.append(f"starts too late ({value.index.min()})")
            if value.index.max() < pd.Timestamp(self.end_date):
                issues.append(f"ends too early ({value.index.max()})")
            if issues:
                problems[name] = "; ".join(issues)

        if problems and raise_error:
            details = "\n".join(
                f"- {name}: {issue}" for name, issue in problems.items()
            )
            message = "Some Simstrat inputs are invalid or incomplete:\n" + details
            self.log.error("%s", message)
            raise ValueError(message)
        if problems:
            self.log.debug("Input validation found problems: %s", problems)
        else:
            self.log.info(
                "All model inputs cover the simulation period %s to %s",
                self.start_date,
                self.end_date,
            )
        return problems

    @staticmethod
    def _read_output_table(
        path: Path,
        *,
        encoding: str,
        preferred_sep: str | None,
    ) -> tuple[pd.DataFrame, str]:
        separators = [preferred_sep, ",", ";", "\t"]
        for separator in dict.fromkeys(item for item in separators if item):
            try:
                frame = pd.read_csv(path, sep=separator, encoding=encoding)
            except (pd.errors.ParserError, UnicodeError):
                continue
            if len(frame.columns) > 1:
                return frame, separator
        raise ValueError(f"Could not detect a tabular separator for {path}.")

    @staticmethod
    def _validate_time_index(index: pd.DatetimeIndex, *, source: Path) -> None:
        if index.hasnans:
            raise ValueError(f"Unparseable timestamps in {source}.")
        if not index.is_monotonic_increasing:
            raise ValueError(f"Timestamps are not sorted in {source}.")
        if index.has_duplicates:
            raise ValueError(f"Duplicate timestamps in {source}.")

    def _output_time_index(
        self,
        values: pd.Series,
        *,
        source: Path,
    ) -> pd.DatetimeIndex:
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.notna().all():
            # Decimal days commonly encode simple intervals such as 30 minutes
            # a few nanoseconds off because of floating-point representation.
            offsets = _timedelta_from_decimal_days(numeric)
            index = pd.Timestamp(self.reference_date) + offsets
        else:
            index = pd.to_datetime(values, errors="coerce")
        result = pd.DatetimeIndex(index, name="time")
        self._validate_time_index(result, source=source)
        return result

    def _normalize_output_depth_columns(
        self,
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        numeric_columns: list[tuple[Any, float]] = []
        for column in frame.columns:
            number = parse_finite_float(column)
            if number is not None:
                numeric_columns.append((column, number))
        if not numeric_columns:
            return frame

        depth_reference = str(
            self.config["Output"].get("OutputDepthReference", "surface")
        ).strip().casefold()
        values = [number for _, number in numeric_columns]

        if depth_reference == "bottom":
            maximum = max(values)
            transformed = [number - maximum for number in values]
        elif depth_reference == "surface":
            transformed = [-abs(number) for number in values]
        else:
            raise SimstratConfigError(
                "Output/OutputDepthReference must be 'surface' or 'bottom', "
                f"got {depth_reference!r}."
            )

        transformed = [0.0 if value == 0 else value for value in transformed]
        if len(set(transformed)) != len(transformed):
            raise ValueError(
                "Depth normalization produced duplicate output columns: "
                f"{transformed}"
            )

        mapping = {
            original: normalized
            for (original, _), normalized in zip(numeric_columns, transformed)
        }
        return frame.rename(columns=mapping)

    def _set_output_depth_metadata(self) -> None:
        candidates: list[list[float]] = []
        for frame in self.outputs.values():
            depths = [
                number
                for column in frame.columns
                if (number := parse_finite_float(column)) is not None
            ]
            if depths:
                candidates.append(depths)
        if not candidates:
            return

        depths = max(candidates, key=len)
        self.depths_output = np.asarray(depths, dtype=float)
        self.depths_output[self.depths_output == 0] = 0
        if self.morphology_datum is not None:
            self.altitudes_output = self.depths_output + self.morphology_datum
            self.depth_to_altitude_table = pd.Series(
                self.altitudes_output,
                index=self.depths_output,
                name="altitude",
            )

    def load_outputs(
        self,
        sep: str | None = None,
        date_col: str = "Datetime",
        resample: str = "",
        *,
        errors: ErrorMode = "raise",
        normalize_depths: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """Load model outputs without rounding or collapsing timestamps."""
        errors = _validate_error_mode(errors)
        self.log.info("Loading model outputs from %s", self.result_path)
        if not self.result_path.is_dir():
            message = f"Simstrat output directory not found: {self.result_path}"
            self.log.error("%s", message)
            raise FileNotFoundError(message)

        self.outputs = {}
        self.output_variables = None
        self.time = None
        self.depths_output = None
        self.altitudes_output = None
        self.depth_to_altitude_table = None
        self.log.info(
            "Ignoring output files %s and suffixes %s",
            sorted(self.output_file_exceptions),
            sorted(self.output_suffix_exceptions),
        )
        for path in sorted(self.result_path.iterdir(), key=lambda p: p.name.casefold()):
            if (
                not path.is_file()
                or path.name in self.output_file_exceptions
                or path.suffix.casefold() in self.output_suffix_exceptions
                or not file_has_content(path)
            ):
                self.log.debug("Skipped output path %s", path)
                continue
            try:
                frame, detected_sep = self._read_output_table(
                    path,
                    encoding=_normalized_encoding(path),
                    preferred_sep=sep,
                )
                frame = self._strip_column_names(frame)

                if sep is not None and detected_sep != sep:
                    self.log.warning(
                        "%s: expected separator %r but read successfully with %r",
                        path.name,
                        sep,
                        detected_sep,
                    )

                if path.name == "_variables.dat":
                    self.output_variables = frame
                    self.log.debug("Loaded output variable metadata from %s", path)
                    continue

                time_column = self._resolve_column(frame, date_col)
                index = self._output_time_index(frame[time_column], source=path)
                frame = frame.drop(columns=time_column)
                frame.index = index

                if normalize_depths:
                    frame = self._normalize_output_depth_columns(frame)
                if resample:
                    from kalden.core.datascience.pandas import df_smart_resample

                    frame = df_smart_resample(frame, resample)
                    self.log.debug("Resampled output %s to %s", path.stem, resample)
                self.outputs[path.stem] = frame
                self.log.debug(
                    "Loaded output %s from %s (%d row(s), %d column(s))",
                    path.stem,
                    path,
                    frame.shape[0],
                    frame.shape[1],
                )
            except Exception as exc:
                self._handle_read_error(path.stem, path, exc, errors)

        self._set_output_depth_metadata()
        self.outputs_loaded = bool(self.outputs)
        indexes = [frame.index for frame in self.outputs.values()]
        if indexes and all(index.equals(indexes[0]) for index in indexes[1:]):
            self.time = indexes[0]
            self.log.debug(
                "All model outputs share %d timestamp(s)",
                len(self.time),
            )
        if self.outputs:
            self.log.info("Loaded %d model output file(s)", len(self.outputs))
        else:
            self.log.warning("No model output files were loaded from %s", self.result_path)
        return self.outputs

    def load_all(
        self,
        inputs_kwargs: Mapping[str, Any] | None = None,
        outputs_kwargs: Mapping[str, Any] | None = None,
        log_kwargs: Mapping[str, Any] | None = None,
    ) -> "SimstratConfig":
        """Load inputs, outputs, and the simulation log."""
        self.log.info("Loading all model data")
        self.load_inputs(**dict(inputs_kwargs or {}))
        self.load_outputs(**dict(outputs_kwargs or {}))
        self.load_log(**dict(log_kwargs or {}))
        self.log.info("Finished loading all model data")
        return self

    def load_log(
        self,
        log_name: str = "simulation_log.log",
        *,
        required: bool = False,
    ) -> str | None:
        """Load the optional simulation log."""
        path = self.root_path / log_name
        self.log.info("Loading simulation log from %s", path)
        if not path.is_file():
            self.simulation_log = None
            if required:
                message = f"Simulation log not found: {path}"
                self.log.error("%s", message)
                raise FileNotFoundError(message)
            self.log.debug("Optional simulation log was not found: %s", path)
            return None
        self.simulation_log = path.read_text(
            encoding=_normalized_encoding(path),
            errors="replace",
        )
        self.log.info("Loaded simulation log from %s", path)
        return self.simulation_log

    def get_execution_date(self) -> datetime | None:
        """Infer the most recent completed execution timestamp."""
        log_match = None
        if self.simulation_log:
            matches = re.findall(
                r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s*End simulation",
                self.simulation_log,
            )
            if matches:
                log_match = pd.Timestamp(matches[-1]).to_pydatetime()

        output_time = None
        if self.result_path.is_dir():
            timestamps = [
                path.stat().st_mtime
                for path in self.result_path.iterdir()
                if path.is_file()
            ]
            if timestamps:
                output_time = datetime.fromtimestamp(max(timestamps))

        self.execution_time = log_match or output_time
        if self.execution_time is not None:
            source = "simulation log" if log_match is not None else "output timestamps"
            self.log.info(
                "Inferred simulation execution time %s from %s",
                self.execution_time,
                source,
            )
        else:
            self.log.warning("Could not infer the simulation execution time")
        return self.execution_time

    def describe(self, *, print_output: bool = True) -> str:
        """Return a short model setup summary."""
        lines = [
            self.par_file,
            f"Simulation start: {self.start_date}",
            f"Simulation end: {self.end_date}",
            f"Simulation timestep [s]: {self.timestep_seconds:g}",
        ]
        if self.execution_time is not None:
            lines.append(f"Simulation execution finished at {self.execution_time}")
        description = "\n".join(lines)
        if print_output:
            print(description)
        return description

    def get_instance_attributes(self) -> list[str]:
        """Return instance attribute names for notebook introspection."""
        attributes = sorted(self.__dict__)
        print(attributes)
        return attributes

    def io_type(self, var_name: str) -> str | None:
        """Return whether a loaded variable is an input or output."""
        if var_name in self.outputs:
            return "output"
        if var_name in self.inputs:
            return "input"
        return None

    @property
    def depth_to_altitude(self) -> pd.Series | None:
        """Compatibility alias for the canonical altitude mapping."""
        return self.depth_to_altitude_table

    @staticmethod
    def df_simstrat_clean_col_name(column: object) -> float:
        """Extract the numeric depth from a Simstrat inflow column name."""
        cleaned = str(column).replace("deep_", "").replace("surface_", "")
        if "_" in cleaned:
            cleaned = cleaned.split("_", 1)[0]
        return float(cleaned)

    @staticmethod
    def df_simstrat_transpose(frame: pd.DataFrame) -> pd.DataFrame:
        """Transpose a time/depth frame and sort depth descending."""
        result = frame.T
        result.index = pd.to_numeric(result.index, errors="raise")
        return result.sort_index(ascending=False)

    @staticmethod
    def df_depths_to_altitudes(
        frame: pd.DataFrame,
        depth_to_altitude: pd.Series,
    ) -> pd.DataFrame:
        """Rename numeric depth columns using an explicit altitude mapping."""
        if depth_to_altitude is None:
            raise ValueError("depth_to_altitude mapping is required.")
        mapping = depth_to_altitude.copy()
        mapping.index = pd.to_numeric(mapping.index, errors="raise")
        columns = pd.Index(pd.to_numeric(frame.columns, errors="raise"))
        missing = columns.difference(mapping.index)
        if not missing.empty:
            raise KeyError(
                f"Missing depth-to-altitude mapping for columns: {list(missing)}"
            )
        result = frame.copy()
        result.columns = mapping.loc[columns].to_numpy()
        return result

    def depths_to_altitudes(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Rename depth columns using this model's altitude mapping."""
        if self.depth_to_altitude_table is None:
            raise ValueError(
                "No altitude mapping is available; set Input/Morphology datum."
            )
        return self.df_depths_to_altitudes(
            frame,
            self.depth_to_altitude_table,
        )


def read_simstrat_model_setup(config_path: PathLike) -> dict[str, Any]:
    """Compatibility wrapper for :meth:`SimstratConfig.read_simstrat_model_setup`."""
    return SimstratConfig.read_simstrat_model_setup(config_path)


def get_simstrat_model_setups(
    root_dir: PathLike,
    exceptions: Sequence[str] | None = None,
) -> list[str]:
    """Compatibility wrapper for :meth:`SimstratConfig.get_simstrat_model_setups`."""
    return SimstratConfig.get_simstrat_model_setups(root_dir, exceptions)


def get_file_path_from_setup(
    setup_path: PathLike,
    keys: Sequence[str],
) -> str:
    """Compatibility wrapper for :meth:`SimstratConfig.get_file_path_from_setup`."""
    return SimstratConfig.get_file_path_from_setup(setup_path, keys)


def compute_sim_dates(
    ref_date: str | datetime | pd.Timestamp,
    operation: str,
    start_date: str | datetime | pd.Timestamp | None = None,
    end_date: str | datetime | pd.Timestamp | None = None,
    day_num: float | None = None,
) -> tuple[int | float, int | float] | pd.Timestamp:
    """Compatibility wrapper for :meth:`SimstratConfig.compute_sim_dates`."""
    return SimstratConfig.compute_sim_dates(
        ref_date,
        operation,
        start_date=start_date,
        end_date=end_date,
        day_num=day_num,
    )
