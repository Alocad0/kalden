# MIKE+ Model Reader

The `MPlusModel` class provides a Python interface for reading and analysing **DHI MIKE+ models** directly from their SQLite databases.

It supports accessing model tables, resolving scenario-specific data, extracting network topology, and performing upstream network analysis without opening MIKE+.

## 1. Open a MIKE+ model

```python
from kalden.core.mike.mike_plus import MPlusModel

# Open a MIKE+ SQLite database
db = MPlusModel(r"C:\Models\Network.sqlite")

# Alternatively, open a MIKE+ project file
db = MPlusModel(r"C:\Models\Network.mupp")
```

When a `.mupp` file is provided, `MPlusModel` resolves the SQLite database using the `DBFilePath` entry in the `[MIKE_URBAN]` section.

The reader uses a temporary SQLite copy where required, avoiding direct modifications to the original model.

## 2. Explore the model database

Inspect the tables available in the model:

```python
# Find scenario-dependent tables
tables = db.scenario_tables()
print(tables)

# Read a standard database table
nodes = db.fetch_table("msm_Node")

print(nodes.head())
print(nodes.columns)
```

`fetch_table()` returns a pandas DataFrame, making it easy to filter records and calculate statistics.

For example:

```python
# Find a particular node
node = nodes.loc[nodes["MUID"] == "Node_001"]

# Inspect node invert levels
print(nodes[["MUID", "InvertLevel"]].head())
```

Actual field names depend on the MIKE+ database schema.

## 3. Work with scenarios

MIKE+ models may contain several scenarios, each representing a different configuration of the network.

`MPlusModel` provides scenario-aware access to database records, including alternative configurations and inherited data.

```python
# Inspect loaded scenario definitions
print(db.scenarios)

# Inspect the currently selected scenario
print(db.scenario)
```

When reading scenario-dependent tables, use the `scenario` argument:

```python
# Read the network nodes for a specific scenario
nodes = db.fetch_table(
    "msm_Node",
    scenario="Scenario_001",
)
```

The reader resolves the applicable records according to the scenario and its alternatives, rather than simply returning every row stored in the SQLite table.

**Note:** Scenario identifiers and the accepted `scenario` argument type should be checked against the installed Kalden version.

## 4. Extract network topology

`fetch_topology()` assembles the network structure, including nodes, links, catchments, and their connections.

```python
topology = db.fetch_topology()

# Inspect the resulting object
print(topology)
```

Topology extraction is useful for network connectivity checks, identifying connected elements, and preparing graph-based analyses.

When supported by the installed version, a specific scenario can be used:

```python
topology = db.fetch_topology(
    scenario="Scenario_001"
)
```

## 5. Upstream network analysis

The `upstream()` method identifies the network elements draining towards a selected node.

```python
# Extract the upstream network of a node
result = db.upstream("Node_001")
```

The result provides access to upstream network elements:

```python
# Upstream nodes
print(result.nodes.head())

# Upstream links
print(result.links.head())

# Upstream catchments
print(result.catchments.head())
```

### Calculate upstream catchment area

When catchments are returned as a GeoDataFrame with a projected coordinate reference system:

```python
# Total upstream catchment area
area_m2 = result.catchments.area.sum()

print(f"Upstream area: {area_m2 / 10_000:.2f} ha")
```

Make sure the geometries use a projected CRS in metres before calculating areas.

### Visualise the upstream network

The analysis result supports interactive mapping using Folium:

```python
# Create an interactive map
m = result.explore()

# Export as standalone HTML
m.save("upstream_network.html")
```

The exported map can be opened in a web browser and shared without requiring MIKE+.

## 6. Typical workflow

```python
from kalden.core.mike.mike_plus import MPlusModel

# Load model
db = MPlusModel("Network.mupp")

# Discover scenario tables
print(db.scenario_tables())

# Extract model data
nodes = db.fetch_table("msm_Node")

# Build network topology
topology = db.fetch_topology()

# Analyse upstream network
result = db.upstream("Node_001")

#
