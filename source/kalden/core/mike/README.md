# MIKE+ tools — `kalden.core.mike`

Read, inspect, compare, and track changes in MIKE+ collection-system (CS) models stored in SQLite/SpatiaLite databases.

This guide covers three complementary modules:

| Module | Main entry point | Purpose |
| --- | --- | --- |
| [`mike_plus.py`](mike_plus.py) | `MPlusModel` | Read `.sqlite` or `.mupp` models, resolve MIKE+ scenarios, extract tables and hydraulic topology, and perform upstream analysis. |
| [`mike_plus_compare.py`](mike_plus_compare.py) | `compare_topologies()` | Compare two **scenario-resolved** topologies, including assets, connectivity, geometry, and selected attributes. |
| [`mike_plus_versioning.py`](mike_plus_versioning.py) | `MikePlusVersioning` | Record **user-triggered checkpoints** of selected **raw SQLite tables**, preserving a change history outside the MIKE+ database. |

**Which tool should I use?** Use `MPlusModel` to answer *what is in this scenario?*; `compare_topologies()` to answer *what differs between these two scenarios or models?*; and `MikePlusVersioning` to answer *what changed in the database since I last checked?*

> **Safety:** These utilities do not intentionally modify the MIKE+ source database. Comparison and model reading operate on a temporary copy. Versioning reads the source in SQLite read-only mode and writes its audit history to a **separate** SQLite file. Check the [database access notes](#database-access-and-important-limitations) before reading an open model.

## Contents

- [Installation and imports](#installation-and-imports)
- [1. Read a MIKE+ model](#1-read-a-mike-model)
- [2. Scenarios and scenario-aware tables](#2-scenarios-and-scenario-aware-tables)
- [3. Extract topology and run upstream analysis](#3-extract-topology-and-run-upstream-analysis)
- [4. Compare scenarios or models](#4-compare-scenarios-or-models)
- [5. Track changes over time (versioning)](#5-track-changes-over-time-versioning)
- [6. Practical end-to-end examples](#6-practical-end-to-end-examples)
- [API quick reference](#api-quick-reference)
- [Database access and important limitations](#database-access-and-important-limitations)

## Installation and imports

From the repository root (Python **3.12+** on the current `dev-codex` branch):

```bash
git clone https://github.com/Alocad0/kalden.git
cd kalden
git checkout dev-codex
python -m pip install -e ".[maps]"
```

The `maps` extra installs `folium` and `mapclassify` for interactive upstream maps. The core dependencies (GeoPandas, pandas, NetworkX, Matplotlib, etc.) are declared in `pyproject.toml`. A separate MIKE+ Python API installation is **not** required to read the SQLite files through these three utilities.

Import directly from each module (the classes/functions are not currently re-exported by `kalden.core.mike.__init__`):

```python
from pathlib import Path

from kalden.core.mike.mike_plus import MPlusModel
from kalden.core.mike.mike_plus_compare import compare_topologies
from kalden.core.mike.mike_plus_versioning import MikePlusVersioning
```

All examples use illustrative paths and scenario names; replace these with values from your own model.

## 1. Read a MIKE+ model

### Open either a `.sqlite` or `.mupp` file

```python
from kalden.core.mike.mike_plus import MPlusModel

# Direct SQLite database
model = MPlusModel(r"C:\models\Project.sqlite")

# Alternatively: MIKE+ project file pointing to that database
model = MPlusModel(r"C:\models\Project.mupp")

print(model.path)     # Path supplied to the constructor
print(model.db_path)  # Resolved path to the SQLite database
```

For `.mupp` files, `MPlusModel` reads `DBFilePath` from the `[MIKE_URBAN]` section and resolves a relative path against the project file directory. The referenced database must be a local `.sqlite` file.

### Discover what the database contains

```python
# All regular table names (printed to the console by default)
model.list_tables()

# Search table names without printing
node_tables = model.list_tables(contains="node", print_results=False)
print(node_tables)

# Include views if required
all_names = model.list_tables(include_views=True, print_results=False)

# Tables with MUID + AltID that can be resolved by scenario
print(model.scenario_tables())
```

### Read tabular data

```python
# Ordinary pandas DataFrame, with MIKE+ alternative resolution
nodes_df = model.fetch_table("msm_Node", scenario="Base")
links_df = model.fetch_table("msm_Link", scenario="Base")
catchment_connections = model.fetch_table("msm_CatchCon", scenario="Base")

print(nodes_df.columns.tolist())
print(nodes_df.head())
print(links_df[["MUID", "FromNodeID", "ToNodeID"]].head())
```

`fetch_table()` returns all selected columns, including raw geometry BLOBs where present. Use `fetch_spatial_table()` for decoded geometries instead.

### Read spatial data (GeoPandas)

```python
nodes = model.fetch_spatial_table("msm_Node", scenario="Base", crs="EPSG:2056")
links = model.fetch_spatial_table("msm_Link", scenario="Base", crs="EPSG:2056")
catchments = model.fetch_spatial_table("msm_Catchment", scenario="Base", crs="EPSG:2056")

print(type(nodes))  # geopandas.GeoDataFrame
print(nodes[["MUID", "geometry"]].head())
print(catchments.geometry.area.sum() / 10_000)  # hectares (metre-based CRS)

# Export a layer for GIS inspection
nodes.to_file("nodes.gpkg", layer="nodes", driver="GPKG")
```

Geometry is decoded in Python directly from SQLite/SpatiaLite blobs: no `mod_spatialite` extension is necessary. The `crs` argument **assigns** the CRS; it does not reproject coordinates. Set it to the actual CRS of your model (typically `EPSG:2056` for Swiss projected models).

## 2. Scenarios and scenario-aware tables

### List scenarios

```python
model = MPlusModel(r"C:\models\Project.mupp")

for scenario in model.scenarios:
    print(
        f"{scenario.name}: MUID={scenario.muid}, "
        f"parent={scenario.parent}, alternatives={scenario.alternatives}"
    )
```

`model.scenarios` contains `MPlusScenario` records with `muid`, `name`, `parent`, and `alternatives`. Use the displayed **scenario name** when fetching data. Pass `scenario="Base"` explicitly for the base version; do not rely on an automatically selected active scenario.

### Read the same table under different scenarios

```python
base = model.fetch_table("msm_Node", scenario="Base")
proposed = model.fetch_table("msm_Node", scenario="Proposed")

print("Base nodes:", len(base))
print("Proposed nodes:", len(proposed))
```

When a table has `MUID` and `AltID`, the reader starts with the Base records (`AltID=0`) and applies the selected scenario's alternative inheritance/overrides and deletions. For other tables, `fetch_table()` returns the table without scenario filtering. This distinction also applies to `msm_CatchCon`: inspect your schema to confirm whether it has the columns needed for scenario resolution.

> **Important:** Querying raw SQLite directly, or simply filtering to a single `AltID`, is **not** equivalent to resolving the complete MIKE+ scenario. Use `fetch_table()`, `fetch_spatial_table()`, or `fetch_topology()` for scenario-specific analysis.

## 3. Extract topology and run upstream analysis

### Load the complete topology

```python
model = MPlusModel(r"C:\models\Project.sqlite")
topology = model.fetch_topology(scenario="Base", crs="EPSG:2056")

print("Nodes:", len(topology.nodes))
print("Links:", len(topology.links))
print("Weirs:", len(topology.weirs))
print("Pumps:", len(topology.pumps))
print("Catchments:", len(topology.catchments))
print("Catchment connections:", len(topology.catchment_connections))
```

`fetch_topology()` returns an `MPlusTopology` object:

| Property | Source table | Type |
| --- | --- | --- |
| `nodes` | `msm_Node` | GeoDataFrame |
| `links` | `msm_Link` | GeoDataFrame |
| `weirs` | `msm_Weir` | GeoDataFrame |
| `pumps` | `msm_Pump` | GeoDataFrame |
| `catchments` | `msm_Catchment` | GeoDataFrame |
| `catchment_connections` | `msm_CatchCon` | DataFrame |
| `graph` | Derived from resolved nodes and links | `networkx.MultiDiGraph` |

By default (`validate=True`), topology extraction checks that every referenced link endpoint exists in the resolved node table. Use `validate=False` only when intentionally investigating inconsistent networks.

### Inspect the network graph

```python
# FromNodeID -> ToNodeID is the directed edge orientation.
print(topology.graph.number_of_nodes())
print(topology.graph.number_of_edges())
print(list(topology.graph.predecessors("N-100")))
print(list(topology.graph.successors("N-100")))
```

The graph currently uses the **links** table for edges. Weirs and pumps are returned as layers but are **not automatically added as graph edges**. An upstream search therefore follows the link direction and may not represent every possible flow path through structures.

### Find everything upstream of a node

```python
upstream = topology.upstream("N-100")

print("Upstream nodes:", len(upstream.nodes))
print("Upstream links:", len(upstream.links))
print("Connected catchments:", len(upstream.catchments))
print(f"Total catchment area: {upstream.total_area_ha:.2f} ha")

# Access the resulting GeoDataFrames for further analysis
upstream.catchments.to_file(
    "upstream_catchments.gpkg", layer="catchments", driver="GPKG"
)
```

`total_area_ha` is calculated from resolved **catchment polygon geometry**, not necessarily a stored `Area` attribute. This assumes a projected CRS expressed in metres (for example `EPSG:2056`). The target node is included in the result.

### Explore the upstream subnetwork on an interactive map

```python
# Requires: pip install -e ".[maps]"
m = upstream.explore()
m.save("upstream_network.html")  # Open the HTML in a browser

# In a Jupyter notebook, simply evaluate m to display the interactive map
m
```

The map shows catchments, links, and nodes, with the target node highlighted. The map is reprojected to `EPSG:4326` for web display. Use `.save("file.html")` to export a Folium map.

## 4. Compare scenarios or models

[`compare_topologies()`](mike_plus_compare.py) compares two `MPlusTopology` objects. Each must first be fetched for the correct model and scenario.

The comparison covers:

- Added, removed, and modified **nodes, links, weirs, pumps, and catchments**, matched by `MUID`.
- Changes to `FromNodeID`/`ToNodeID` for links (and for weirs/pumps when those fields exist).
- Geometry differences (with optional spatial tolerance).
- **Catchment connections**, matched by `CatchID` and compared as a multiset of assigned `NodeID` values.
- Optional scalar attributes, selected explicitly (e.g. invert level, diameter).

It does **not** yet perform a generic comparison of boundary conditions or every field in the model.

### Example A — Compare two scenarios in the same database

```python
from kalden.core.mike.mike_plus import MPlusModel
from kalden.core.mike.mike_plus_compare import compare_topologies

model = MPlusModel(r"C:\models\Project.sqlite")

reference = model.fetch_topology(scenario="Base")
candidate = model.fetch_topology(scenario="Proposed")

comparison = compare_topologies(reference, candidate)

print(comparison.summary.to_string(index=False))
print(comparison.changes.head(20))
```

`comparison.summary` counts assets before/after and gives `added`, `removed`, `modified`, and `unchanged` totals per layer. `comparison.changes` lists field-level changes with the columns `layer`, `asset_id`, `change`, `field`, `before`, and `after`.

```python
# View only changes affecting links
link_changes = comparison.for_layer("links")

# View catchments redirected to different nodes
connection_changes = comparison.for_layer("catchment_connections")
print(connection_changes[["asset_id", "change", "before", "after"]])
```

### Example B — Compare two different SQLite models

```python
old_model = MPlusModel(r"C:\models\Model_2025.sqlite")
new_model = MPlusModel(r"C:\models\Model_2026.sqlite")

old_topology = old_model.fetch_topology(scenario="Base", crs="EPSG:2056")
new_topology = new_model.fetch_topology(scenario="Base", crs="EPSG:2056")

comparison = compare_topologies(old_topology, new_topology)
print(comparison.summary)
```

Both topologies must have defined and matching CRSs. The comparison is purely structural: it does not require the scenarios to have the same name.

### Example C — Also compare hydraulic attributes

Scalar attributes are **opt-in**; otherwise the tool focuses on entities, connections, and geometry. First inspect the columns exposed by your MIKE+ schema:

```python
print(reference.nodes.columns.tolist())
print(reference.links.columns.tolist())
```

Then choose columns that actually exist **in both versions**:

```python
comparison = compare_topologies(
    reference,
    candidate,
    attributes={
        "nodes": ["InvertLevel"],  # Example: verify this column exists
        "links": ["Diameter"],     # Example: verify this column exists
    },
    geometry_tolerance=0.01,  # 1 cm in a metre-based CRS
    numeric_tolerance=1e-3,   # Absolute tolerance, in each attribute's units
)

print(comparison.for_layer("nodes"))
print(comparison.for_layer("links"))
```

The library raises an error for unavailable attribute columns rather than silently ignoring them. `geometry_tolerance` and `numeric_tolerance` must be non-negative. Geometry is compared automatically and must not be added to `attributes`.

### Export the comparison for reporting or QGIS

```python
from pathlib import Path

out = Path("output/model_comparison")
comparison.to_csv(out)  # Creates summary.csv and changes.csv

# Changed spatial assets only: nodes, links, weirs, pumps, catchments
written_layers = comparison.to_geopackage(out / "changes.gpkg")
print("GeoPackage layers:", written_layers)
```

The GeoPackage records the change type, original geometry, and whether each geometry belongs to the `before` or `after` version; modified features appear on **both** sides. Catchment connections are exported to CSV only, not a separate spatial layer. `to_geopackage()` **refuses to overwrite an existing output file**, and may produce no file if no changed spatial geometries exist.

## 5. Track changes over time (versioning)

`MikePlusVersioning` is a lightweight audit tool, not a complete MIKE+ edit log. Each call to `check()` scans the selected **raw SQLite tables**, compares them with their previous stored state, and records row/field changes in a **separate history SQLite database**.

- The first check establishes a **baseline** and reports **no changes**.
- Later checks report `INSERT`, `DELETE`, and field-level `UPDATE` events.
- Events are dated by the **checkpoint time**, not the unknown exact time of the MIKE+ edit.
- The default tracked tables are `msm_Node` and `msm_Link`.
- Tracking is at the **physical table row** level (including alternatives), **not** at the resolved-scenario level.

### Step 1 — Create an audit and baseline

```python
from kalden.core.mike.mike_plus import MPlusModel
from kalden.core.mike.mike_plus_versioning import MikePlusVersioning

model = MPlusModel(r"C:\models\Project.mupp")

audit = MikePlusVersioning(
    model.db_path,                   # Actual .sqlite path (not the .mupp path)
    # history_path can be omitted to use the default .kalden/ location
    tables=("msm_Node", "msm_Link"),
)

baseline_changes = audit.check(note="Initial model baseline")
print(baseline_changes)  # [] — first check establishes the baseline
print(audit.history_path)  # Separate .kalden/Project.history.sqlite
```

**Recommended workflow:** Create the baseline *before* making the next set of edits in MIKE+.

### Step 2 — Recheck after editing the model

After changes have been **saved** in MIKE+, run:

```python
new_changes = audit.check(note="After design adjustments")

for change in new_changes:
    print(
        change.checkpoint_id,
        change.table,
        change.row_key,
        change.operation,
        change.column,
        change.old_value,
        "->",
        change.new_value,
    )
```

Each `Change` record exposes `checkpoint_id`, `table`, `row_key`, `operation`, `column`, `old_value`, and `new_value`. A `row_key` is a serialized, type-aware representation of the table's primary key (or `MUID` fallback); it is not necessarily a plain asset ID.

### Step 3 — Inspect previous checks and changes

```python
# List checks, newest first: id, checked_at (UTC), note, changes
for checkpoint in audit.checkpoints():
    print(checkpoint)

# All recorded events
events = audit.changes()

# Changes from the latest checkpoint only
latest_id = audit.checkpoints()[0]["id"]
latest_events = audit.changes(checkpoint_id=latest_id)

# Restrict to one table
node_events = audit.changes(table="msm_Node")

# Filter by exact serialized row key from a previous event
if node_events:
    asset_history = audit.changes(table="msm_Node", row_key=node_events[0].row_key)
    print(asset_history)
```

The API also allows combinations of `checkpoint_id=`, `table=`, and `row_key=` when retrieving events.

### Tracking more tables

```python
# Example only: confirm that these tables exist and have suitable unique keys.
audit_extended = MikePlusVersioning(
    model.db_path,
    history_path="output/extended_model_history.sqlite",
    tables=("msm_Node", "msm_Link", "msm_Weir", "msm_Pump"),
)
audit_extended.check(note="Extended baseline")
```

The tracked table list is fixed for each history database: **changing `tables` requires a new `history_path`**. A tracked table must have a declared primary key or a unique `MUID`; if `MUID` repeats across alternatives without a composite primary key, the current versioning implementation can fail on duplicate keys. Verify your schema before extending tracking.

> **Versioning is not Git.** It does not reconstruct intermediate edits that occurred between checks, infer who changed a value, or compare effective MIKE+ scenarios. For a scenario-aware comparison, use `compare_topologies()`.

## 6. Practical end-to-end examples

### A. Daily notebook: inspect model, selected scenario, and upstream area

```python
from kalden.core.mike.mike_plus import MPlusModel

model = MPlusModel(r"C:\models\Project.mupp")
print([scenario.name for scenario in model.scenarios])

scenario_name = "Proposed"  # Replace with a name from the list above
topo = model.fetch_topology(scenario=scenario_name)

result = topo.upstream("N-100")  # Replace with a real node MUID
print(f"{len(result.catchments)} catchments, {result.total_area_ha:.2f} ha")

result.explore().save("output/upstream_proposed.html")
```

Create the `output/` directory first if it does not already exist; Folium does not create missing parent directories.

### B. Design review: compare models and export changes

```python
from pathlib import Path
from kalden.core.mike.mike_plus import MPlusModel
from kalden.core.mike.mike_plus_compare import compare_topologies

before = MPlusModel(r"C:\models\Previous.sqlite").fetch_topology("Base")
after = MPlusModel(r"C:\models\Current.sqlite").fetch_topology("Base")

report = compare_topologies(before, after, geometry_tolerance=0.01)

folder = Path("output/design_review")
report.to_csv(folder)
report.to_geopackage(folder / "changed_assets.gpkg")
print(report.summary.to_string(index=False))
```

### C. Simple audit loop

```python
from kalden.core.mike.mike_plus_versioning import MikePlusVersioning

history = MikePlusVersioning(r"C:\models\Project.sqlite")

# Run manually after a saved design session (or the first time, to baseline).
events = history.check(note="Design review 2026-10-23")
print(f"{len(events)} changes recorded")
```

The history file persists across Python sessions. Keep it backed up alongside your project documentation (but **not** inside the original MIKE+ database).

## API quick reference

| Class / function | Signature / key arguments | Result |
| --- | --- | --- |
| `MPlusModel(path)` | `.sqlite` or `.mupp` path | Open model descriptor; `db_path` points to SQLite. |
| `model.scenarios` | property | Tuple of `MPlusScenario` records. |
| `model.list_tables()` | `contains`, `include_views`, `print_results` | List of table/view names. |
| `model.scenario_tables()` | no arguments | Tables containing `MUID` and `AltID`. |
| `model.fetch_table(name)` | `scenario=None` | Scenario-resolved `pandas.DataFrame`. |
| `model.fetch_spatial_table(name)` | `geometry_column="Geometry"`, `crs="EPSG:2056"`, `scenario=None` | Decoded GeoDataFrame. |
| `model.fetch_topology()` | `scenario="Base"`, `crs="EPSG:2056"`, `validate=True` | `MPlusTopology`. |
| `topology.upstream(node_id)` | node MUID | `UpstreamResult`. |
| `upstream.explore()` | no arguments | Folium map. |
| `compare_topologies(before, after)` | `attributes=None`, `geometry_tolerance=0.0`, `numeric_tolerance=1e-9` | `MPlusComparison`. |
| `comparison.for_layer(layer)` | e.g. `"links"` | Filtered change DataFrame. |
| `comparison.to_csv(directory)` | output folder | `summary.csv`, `changes.csv`. |
| `comparison.to_geopackage(path)` | **new** `.gpkg` path | List of layers written. |
| `MikePlusVersioning(db_path)` | `history_path=None`, `tables=("msm_Node", "msm_Link")` | Persistent audit helper. |
| `audit.check(note="")` | optional checkpoint label | List of new `Change` records. |
| `audit.checkpoints()` | no arguments | List of checkpoint dictionaries. |
| `audit.changes()` | `checkpoint_id=None`, `table=None`, `row_key=None` | List of historical `Change` records. |

## Database access and important limitations

1. **Read models when MIKE+ is closed or the database is stable.** `MPlusModel` copies the SQLite file and refuses to read when `-wal`, `-shm`, or `-journal` sidecar files exist. This prevents an inconsistent file-only copy, but it means the read operation may fail while MIKE+ has the database open. Save and close the model first; do not manually delete active sidecar files.
2. **Versioning uses a different snapshot mechanism.** `MikePlusVersioning` opens the source read-only and uses SQLite's backup API, which can include committed WAL changes. It is therefore different from `MPlusModel`'s file-copy strategy. Saved/committed changes are still necessary for the audit to observe them.
3. **Scenario comparison and versioning answer different questions.** `compare_topologies()` resolves alternatives and compares only supported topology layers plus chosen attributes. Versioning scans raw SQLite records on selected tables, including `AltID` variants, and does not know which scenario's effective topology changed.
4. **MUID identity and row keys.** Comparison matches spatial assets by `MUID` and catchment connections by `CatchID`; identifiers must be valid/unique after scenario resolution. Versioning uses SQLite primary keys where available or `MUID` as a fallback, and rejects duplicates.
5. **CRS and units matter.** Geometry tolerances use coordinate-system units. Catchment area calculations assume metres. Set the actual CRS when loading data; attaching an incorrect CRS does not convert coordinates.
6. **The upstream graph uses links.** The `MultiDiGraph` is directed using `FromNodeID` → `ToNodeID` and does not incorporate weirs/pumps as edges. The computed contributing area follows those link connections, not a dynamic hydraulic simulation.
7. **Change history is sampled, not continuous.** Checkpoint timestamps indicate when a difference was **observed**, not when an operator made it; several changes between checks collapse into the final state.
8. **Large models can take time.** Model reading copies the database for each read. Versioning scans and hashes entire selected raw tables at each check, including any stored geometry BLOBs. Use selected tables and schedule checks appropriately.
9. **No boundary-condition diff yet.** The current comparison implementation targets CS topology; it is not a complete model-settings or boundary-condition diff engine.

### Relevant source files

- [`mike_plus.py`](mike_plus.py) — scenario-aware model reader, topology, and upstream analysis.
- [`mike_plus_compare.py`](mike_plus_compare.py) — topology/attribute comparison and CSV/GeoPackage export.
- [`mike_plus_versioning.py`](mike_plus_versioning.py) — separate, checkpoint-based change history.
- [`spatialite_blob.py`](spatialite_blob.py) — low-level SpatiaLite geometry decoding used by the reader.

**Target location:** `source/kalden/core/mike/README.md` on `dev-codex`.
