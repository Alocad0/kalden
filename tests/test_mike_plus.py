import sqlite3
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, Polygon

from kalden.core.mike.mike_plus import MPlusModel, copied_sqlite_connection


def _create_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE Alpha (id INTEGER PRIMARY KEY, value TEXT)")
        connection.execute("INSERT INTO Alpha (value) VALUES ('original')")
        connection.execute("CREATE TABLE beta (id INTEGER PRIMARY KEY)")
        connection.execute("CREATE VIEW alpha_view AS SELECT * FROM Alpha")
        connection.commit()
    finally:
        connection.close()


def test_copied_sqlite_connection_is_read_only_and_preserves_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "model.sqlite"
    _create_database(source)

    with copied_sqlite_connection(source) as connection:
        assert connection.execute("SELECT value FROM Alpha").fetchone() == (
            "original",
        )
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("INSERT INTO Alpha (value) VALUES ('changed')")

    connection = sqlite3.connect(source)
    try:
        assert connection.execute("SELECT value FROM Alpha").fetchall() == [
            ("original",)
        ]
    finally:
        connection.close()


@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_copied_sqlite_connection_rejects_active_sidecars(
    tmp_path: Path,
    suffix: str,
) -> None:
    source = tmp_path / "model.sqlite"
    _create_database(source)
    Path(f"{source}{suffix}").touch()

    with pytest.raises(RuntimeError, match="Sidecar files found"):
        with copied_sqlite_connection(source):
            pass


def test_list_tables_filters_names_and_optionally_includes_views(
    tmp_path: Path,
) -> None:
    source = tmp_path / "model.sqlite"
    _create_database(source)
    model = MPlusModel(source)

    assert model.list_tables(print_results=False) == ["Alpha", "beta"]
    assert model.list_tables(contains="ALPHA", print_results=False) == ["Alpha"]
    assert model.list_tables(include_views=True, print_results=False) == [
        "Alpha",
        "alpha_view",
        "beta",
    ]


def test_sqlite_identifier_quoting_and_column_resolution() -> None:
    assert MPlusModel._quote_identifier('table"name') == '"table""name"'
    with pytest.raises(ValueError, match="non-empty strings"):
        MPlusModel._quote_identifier("")

    frame = pd.DataFrame(columns=["MUID", "value"])
    assert MPlusModel._resolve_column(frame, "muid") == "MUID"

    ambiguous = pd.DataFrame(columns=["MUID", "muid"])
    with pytest.raises(ValueError, match="Multiple columns"):
        MPlusModel._resolve_column(ambiguous, "muid")


def test_build_link_geometries_uses_endpoint_points_and_crs() -> None:
    nodes = gpd.GeoDataFrame(
        {"MUID": ["N1", "N2", "N3"]},
        geometry=[Point(0, 0), Point(2, 0), Point(2, 3)],
        crs="EPSG:2056",
    )
    links = pd.DataFrame(
        {
            "FromNodeID": ["N1", "N2"],
            "ToNodeID": ["N2", "N3"],
            "name": ["L1", "L2"],
        }
    )

    result = MPlusModel.build_link_geometries_from_nodes(nodes, links)

    assert isinstance(result, gpd.GeoDataFrame)
    assert result.crs == nodes.crs
    assert result["name"].tolist() == ["L1", "L2"]
    assert list(result.geometry.iloc[0].coords) == [(0.0, 0.0), (2.0, 0.0)]
    assert list(result.geometry.iloc[1].coords) == [(2.0, 0.0), (2.0, 3.0)]


def test_build_link_geometries_rejects_missing_endpoint() -> None:
    nodes = gpd.GeoDataFrame(
        {"MUID": ["N1"]},
        geometry=[Point(0, 0)],
        crs="EPSG:2056",
    )
    links = pd.DataFrame({"FromNodeID": ["N1"], "ToNodeID": ["missing"]})

    with pytest.raises(ValueError, match="reference missing node geometry"):
        MPlusModel.build_link_geometries_from_nodes(nodes, links)


def test_build_catchment_connection_geometry_uses_centroid() -> None:
    catchment = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    row = pd.Series({"catchment": catchment, "node": Point(4, 1)})

    geometry = MPlusModel.build_catchment_connection_geometry(
        row,
        "catchment",
        "node",
    )

    assert isinstance(geometry, LineString)
    assert list(geometry.coords) == [(1.0, 1.0), (4.0, 1.0)]


def test_validate_catchment_connections_accepts_one_to_one_mapping() -> None:
    catchments = pd.DataFrame({"MUID": ["C1", "C2"]})
    connections = pd.DataFrame({"CatchID": ["C2", "C1"]})

    assert MPlusModel.validate_catchment_connections(catchments, connections)


def test_validate_catchment_connections_reports_all_mapping_problems() -> None:
    catchments = pd.DataFrame({"MUID": ["C1", "C2", "C2", None]})
    connections = pd.DataFrame({"CatchID": ["C1", "C1", "C3", None]})

    with pytest.raises(ValueError) as exc_info:
        MPlusModel.validate_catchment_connections(catchments, connections)

    message = str(exc_info.value)
    assert "null 'MUID'" in message
    assert "null 'CatchID'" in message
    assert "Duplicate catchment identifiers: C2" in message
    assert "Catchments with multiple connections: C1 (2)" in message
    assert "Catchments without a connection: C2" in message
    assert "unknown catchments: C3" in message
