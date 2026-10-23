"""Drop this test into kalden's pytest tests directory."""
from types import SimpleNamespace

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, Polygon

from kalden.core.mike.mike_plus_compare import compare_topologies


def spatial(rows, geometries):
    return gpd.GeoDataFrame(rows, geometry=geometries, crs="EPSG:2056")


def build_topologies():
    before = SimpleNamespace(
        scenario="Base",
        nodes=spatial(
            {"MUID": ["N1", "N2"], "InvertLevel": [100.0, 90.0], "AltID": [0, 0]},
            [Point(0, 0), Point(10, 0)],
        ),
        links=spatial(
            {"MUID": ["L1", "L2"], "FromNodeID": ["N1", "N1"], "ToNodeID": ["N2", "N2"]},
            [LineString([(0, 0), (10, 0)]), LineString([(0, 1), (10, 1)])],
        ),
        weirs=spatial({"MUID": ["W1"], "FromNodeID": ["N1"], "ToNodeID": ["N2"]}, [Point(5, 0)]),
        pumps=spatial({"MUID": ["P1"], "FromNodeID": ["N1"], "ToNodeID": ["N2"]}, [Point(6, 0)]),
        catchments=spatial({"MUID": ["C1"]}, [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]),
        catchment_connections=pd.DataFrame({"CatchID": ["C1", "C1"], "NodeID": ["N1", "N2"]}),
    )
    after = SimpleNamespace(
        scenario="Alternative 1",
        nodes=spatial(
            {"MUID": ["N1", "N2", "N3"], "InvertLevel": [99.9, 90.0, 85.0], "AltID": [12, 12, 12]},
            [Point(0.1, 0), Point(10, 0), Point(20, 0)],
        ),
        links=spatial(
            {"MUID": ["L1", "L3"], "FromNodeID": ["N1", "N2"], "ToNodeID": ["N3", "N3"]},
            [LineString([(0, 0), (20, 0)]), LineString([(10, 0), (20, 0)])],
        ),
        weirs=spatial({"MUID": ["W1"], "FromNodeID": ["N1"], "ToNodeID": ["N3"]}, [Point(5, 0)]),
        pumps=spatial({"MUID": []}, []),
        catchments=spatial({"MUID": ["C1"]}, [Polygon([(0, 0), (2, 0), (2, 1), (0, 1)])]),
        catchment_connections=pd.DataFrame({"CatchID": ["C1", "C1"], "NodeID": ["N1", "N3"]}),
    )
    return before, after


def test_compare_topology_changes():
    before, after = build_topologies()
    result = compare_topologies(before, after)
    summary = result.summary.set_index("layer")
    assert summary.loc["nodes", "added"] == 1
    assert summary.loc["nodes", "modified"] == 1
    assert summary.loc["links", "added"] == 1
    assert summary.loc["links", "removed"] == 1
    assert summary.loc["links", "modified"] == 1
    assert summary.loc["weirs", "modified"] == 1
    assert summary.loc["pumps", "removed"] == 1
    assert summary.loc["catchments", "modified"] == 1
    assert summary.loc["catchment_connections", "modified"] == 1
    fields = result.changes.query("layer == 'nodes' and asset_id == 'N1'")["field"].to_list()
    assert fields == ["geometry"]  # AltID and elevation excluded by default
    connection = result.for_layer("catchment_connections").iloc[0]
    assert connection["before"] == ("N1", "N2")
    assert connection["after"] == ("N1", "N3")
    assert set(result.spatial["nodes"].query("asset_id == 'N1'")["side"]) == {"before", "after"}


def test_opt_in_attributes_tolerance():
    before, after = build_topologies()
    result = compare_topologies(before, after, attributes={"nodes": ["invertlevel"]}, numeric_tolerance=0.01)
    fields = result.changes.query("layer == 'nodes' and asset_id == 'N1'")["field"].to_list()
    assert "invertlevel" in fields and "geometry" in fields
    result2 = compare_topologies(before, after, attributes={"nodes": ["InvertLevel"]}, numeric_tolerance=0.2)
    assert result2.changes.query("layer == 'nodes' and asset_id == 'N1'")["field"].tolist() == ["geometry"]


def test_bad_duplicate_ids_and_crs():
    before, after = build_topologies()
    before.nodes = pd.concat([before.nodes, before.nodes.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate MUID"):
        compare_topologies(before, after)
    before, after = build_topologies()
    after.nodes = after.nodes.to_crs("EPSG:4326")
    with pytest.raises(ValueError, match="same defined CRS"):
        compare_topologies(before, after)


def test_exports(tmp_path):
    result = compare_topologies(*build_topologies())
    result.to_csv(tmp_path / "csv")
    assert (tmp_path / "csv/changes.csv").exists()
    layers = result.to_geopackage(tmp_path / "compare.gpkg")
    assert "nodes" in layers and "catchments" in layers
    assert len(gpd.read_file(tmp_path / "compare.gpkg", layer="nodes")) == 3
    assert len(gpd.read_file(tmp_path / "compare.gpkg", layer="catchments")) == 2
    with pytest.raises(FileExistsError):
        result.to_geopackage(tmp_path / "compare.gpkg")


def test_same_topology_is_unchanged():
    before, _ = build_topologies()
    result = compare_topologies(before, before)
    assert result.changes.empty
    assert (result.summary[["added", "removed", "modified"]] == 0).all().all()
    assert result.to_geopackage  # Export API is available without I/O side effects.
