# kalden

`kalden` is a small Python utilities package that provides helpers for
filesystem operations, data handling, MIKE/Geo workflows, and Simstrat model
I/O.

## Features

- Simple filesystem helpers for directory creation, cleanup, and temporary directories
- File existence and encoding detection utilities
- Data science helpers for pandas workflows and time series processing
- Geospatial helpers for reading/writing spatial files and styling QGIS layers
- MIKE-specific utilities for `dfs0` and `res1d` workflows
- Strict readers for physical Simstrat and Simstrat-SELMA model setups

## Installation

Install from source using:

```bash
pip install .
```

For development dependencies, install:

```bash
pip install .[dev]
```

Optional features:

```bash
pip install .[plot-export]
pip install .[mikeplus]
pip install .[widgets]
pip install .[maps]
```

## Quick Start

```python
from kalden.core.io import hello, create_temp_dir

print(hello("World"))

temp_dir = create_temp_dir()
print(f"Temporary directory created at: {temp_dir}")
```

Read a Simstrat model setup and its tabular data with timestamped standard
logging enabled:

```python
from kalden.core.simstrat import SimstratConfig, configure_simstrat_logging

configure_simstrat_logging()  # INFO by default; call once per process/notebook.
model = SimstratConfig("path/to/model.par")
inputs = model.load_inputs()
outputs = model.load_outputs(sep=",")
```

The logger hierarchy starts at `kalden.core.simstrat`. Applications that already
configure Python logging should configure that logger or the root logger instead
of calling `configure_simstrat_logging()`.

Generate and atomically replace a live Simstrat inflow file without forcing
all values to two decimals:

```python
from kalden.core.simstrat import write_inflow_file

write_inflow_file(
    "path/to/Qin.dat",
    flow_dataframe,
    model.reference_date,
    deep_flows=[{"depth": -2, "col": "Q", "header": "Q [m3/s]"}],
    surface_flows=[],
)
```

Proposed notebook replacements using these APIs are available in:

- `examples/notebooks/Simstrat_inputs_v2.ipynb`
- `examples/notebooks/Simstrat_outputs_analysis_v2.ipynb`

## Project Structure

- `source/kalden/`
  - `core/` — core utilities and IO helpers
  - `core/datascience/` — pandas and generic data helpers
  - `core/spatial/` — geospatial IO and QGIS style helpers
  - `core/mike/` — MIKE format and time series conversion utilities
  - `core/simstrat/` — Simstrat and Simstrat-SELMA setup and table readers
  - `misc/` — plotting utilities
  - `styles/` — bundled QML styles
  - `templates/` — package templates

## Tests

The repository uses `pytest` for unit tests. Run the test suite with:

```bash
python -m pytest
```

## License

This project is released under the terms of the `LICENSE` file.

