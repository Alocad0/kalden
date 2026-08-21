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

Read a Simstrat model setup and its tabular data with:

```python
from kalden.core.simstrat import SimstratConfig

model = SimstratConfig("path/to/model.par")
inputs = model.load_inputs()
outputs = model.load_outputs(sep=",")
```

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

