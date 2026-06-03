# kalden

`kalden` is a small utility library for file I/O, data science helpers, MIKE (mikeio) timeseries workflows, and geospatial exports.

## Features

- `kalden.core.io`
  - file and directory helpers
  - temporary directory creation and cleanup
  - file encoding detection
- `kalden.core.datascience.pandas`
  - pandas Series/DataFrame utility helpers
  - gap filling and resampling helpers
  - EUM type/unit catalog helpers for MIKE IO
- `kalden.core.mike.timeseries`
  - convenient MIKE dfs0 file reading and validation
  - dfs0 rewrite and conversion helpers
  - duplicate timestamp scanning and file iteration
- `kalden.core.spatial`
  - DXF geometry extraction
  - GeoPackage / shapefile export and QML style insertion
- `kalden.misc.plotting`
  - Plotly figure export helpers
  - heatmap color scale utilities

## Installation

Install from source with a Python 3.10+ interpreter:

```bash
python -m pip install .
```

For development and testing, install optional dev dependencies:

```bash
python -m pip install .[dev]
```

If you want plot export support:

```bash
python -m pip install .[plot-export]
```

For MIKE+ support, install the extra dependency group:

```bash
python -m pip install .[mikeplus]
```

## Quick start

```python
from kalden import hello
from kalden.core.io import ensure_dir_exists
from kalden.core.mike.timeseries import Dfs0

print(hello("World"))

ensure_dir_exists("output")
reader = Dfs0("data/sample.dfs0")
df = reader.to_dataframe()
print(df.head())
```

## Example usage

### File system helpers

```python
from kalden.core.io import ensure_file_dir_exists, empty_dir
ensure_file_dir_exists("output/report.txt")
empty_dir("temp", missing_ok=True)
```

### MIKE dfs0 helpers

```python
from kalden.core.mike.timeseries import Dfs0
reader = Dfs0("path/to/file.dfs0")
dataset = reader.validate_timestamps()
converted = reader.rewrite("path/to/output.dfs0", overwrite=True)
```

### Export spatial data

```python
import geopandas as gpd
from kalden.core.spatial.io import export_gdf

gdf = gpd.GeoDataFrame(...)
export_gdf(gdf, "output/layers.gpkg", layer_name="my_layer", overwrite=True)
```

## Repository structure

- `source/kalden/` - main package source code
- `tests/` - unit tests
- `pyproject.toml` - packaging configuration

## Notes

This repository is designed as a lightweight utility package. The MIKE workflows require the `mikeio` package, and geospatial workflows depend on `geopandas`, `fiona`, and `shapely`.

