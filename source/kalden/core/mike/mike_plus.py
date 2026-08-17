"""
Utilities for reading and analysing MIKE+ model databases.

The module provides a small wrapper around MIKE+ SQLite/SpatiaLite databases,
along with helpers for constructing and validating network geometries.

Author: DEAO
Created: 2026-01-15
"""

from __future__ import annotations

import sqlite3
import shutil
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator
from contextlib import contextmanager

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from shapely.geometry import LineString
from shapely.wkt import loads
from tqdm.auto import tqdm


@contextmanager
def copied_sqlite_connection(
    source_path: str | Path,
) -> Iterator[sqlite3.Connection]:
    """
    Copy a closed SQLite database to a temporary directory and open only
    the temporary copy.

    The original file is never opened through SQLite.
    """
    source = Path(source_path).expanduser().resolve(strict=True)

    if not source.is_file():
        raise ValueError(f"Not a regular file: {source}")

    wal = Path(f"{source}-wal")
    shm = Path(f"{source}-shm")
    journal = Path(f"{source}-journal")

    active_sidecars = [
        path
        for path in (wal, shm, journal)
        if path.exists()
    ]

    if active_sidecars:
        formatted_paths = ", ".join(str(path) for path in active_sidecars)
        raise RuntimeError(
            "The database may be active or require recovery. "
            f"Sidecar files found: {formatted_paths}"
        )

    with TemporaryDirectory(prefix="kalden-mikeplus-") as temporary_directory:
        copied_database = (
            Path(temporary_directory) / source.name
        )

        # Ordinary operating-system file copy. SQLite never sees the source.
        shutil.copy2(source, copied_database)

        copied_uri = (
            f"{copied_database.resolve().as_uri()}"
            "?mode=ro&immutable=1"
        )

        connection = sqlite3.connect(
            copied_uri,
            uri=True,
            isolation_level=None,
        )

        try:
            connection.execute("PRAGMA query_only = ON")
            yield connection
        finally:
            connection.close()


class MPlusModel:
    """Read and analyse content from a MIKE+ SQLite database."""

    def __init__(self, db_path: str | PathLike[str]) -> None:
        """
        Initialise the MIKE+ model helper.

        Args:
            db_path: Path to the MIKE+ SQLite database file.
        """
        self.db_path = Path(db_path).expanduser()

    @staticmethod
    def _quote_identifier(identifier: str) -> str:
        """Safely quote an SQLite table or column identifier."""
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("SQLite identifiers must be non-empty strings.")

        return '"' + identifier.replace('"', '""') + '"'

    @staticmethod
    def _resolve_column(
        dataframe: pd.DataFrame,
        requested_column: str,
    ) -> Any:
        """Resolve a DataFrame column name case-insensitively."""
        matches = [
            column
            for column in dataframe.columns
            if str(column).casefold() == requested_column.casefold()
        ]

        if not matches:
            raise ValueError(
                f"Column '{requested_column}' was not found. "
                f"Available columns: {list(dataframe.columns)}"
            )

        if len(matches) > 1:
            raise ValueError(
                f"Multiple columns match '{requested_column}': {matches}"
            )

        return matches[0]

    @staticmethod
    def _require_active_geometry(gdf: gpd.GeoDataFrame, name: str) -> None:
        """Validate that a GeoDataFrame has an active geometry column."""
        if not isinstance(gdf, gpd.GeoDataFrame):
            raise TypeError(f"{name} must be a geopandas.GeoDataFrame.")

        try:
            gdf.geometry
        except AttributeError as exc:
            raise ValueError(
                f"{name} does not have an active geometry column."
            ) from exc

    @staticmethod
    def _require_projected_crs(gdf: gpd.GeoDataFrame, purpose: str) -> None:
        """Require a projected CRS before calculating area or length."""
        if gdf.crs is None:
            raise ValueError(
                f"The GeoDataFrame CRS must be set before calculating {purpose}."
            )

        if gdf.crs.is_geographic:
            raise ValueError(
                f"A projected CRS is required for {purpose}; received {gdf.crs}."
            )

    def list_tables(
        self,
        contains: str | None = None,
        *,
        include_views: bool = False,
        print_results: bool = True,
    ) -> list[str]:
        """
        List tables available in the MIKE+ database.

        Args:
            contains:
                Optional case-insensitive substring used to filter names.
            include_views:
                Whether to include database views.
            print_results:
                Whether to print each matching name.

        Returns:
            Sorted list of matching table and optionally view names.
        """
        object_types = ("table", "view") if include_views else ("table",)
        placeholders = ", ".join("?" for _ in object_types)

        query = f"""
            SELECT name
            FROM sqlite_master
            WHERE type IN ({placeholders})
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name COLLATE NOCASE;
        """

        # with sqlite3.connect(str(self.db_path)) as connection:
        #     rows = connection.execute(query, object_types).fetchall()
        
        # Safer connection to Database - must ensure there is no concurrent use of database (incomplete data)
        with copied_sqlite_connection(self.db_path) as connection:
            rows = connection.execute(query, object_types).fetchall()

        names = [row[0] for row in rows]

        if contains:
            search_value = contains.casefold()
            names = [
                name
                for name in names
                if search_value in name.casefold()
            ]

        if print_results:
            for name in names:
                print(name)

            print(f"\n{len(names)} table(s) found.")

        return names

    def fetch_table_attributes_geometry(
        self,
        table_name: str,
        geometry_column: str = "Geometry",
        crs: str = "EPSG:2056",
    ) -> gpd.GeoDataFrame:
        """
        Fetch every attribute and the geometry from a spatial MIKE+ table.

        Args:
            table_name:
                Name of the spatial table or view to query.
            geometry_column:
                Name of the SpatiaLite geometry column.
            crs:
                CRS assigned to the returned GeoDataFrame.

        Returns:
            GeoDataFrame containing every non-geometry attribute and a Shapely
            geometry column.

        Raises:
            ValueError:
                If the table or geometry column does not exist.
            sqlite3.Error:
                If the database query fails.
            RuntimeError:
                If the SpatiaLite extension cannot be loaded.
        """
        # with sqlite3.connect(str(self.db_path)) as connection:
        #     connection.enable_load_extension(True)

        #     try:
        #         connection.execute('SELECT load_extension("mod_spatialite")')
        #     except sqlite3.Error as exc:
        #         raise RuntimeError(
        #             "Could not load the 'mod_spatialite' SQLite extension."
        #         ) from exc

        #     table_row = connection.execute(
        #         """
        #         SELECT name
        #         FROM sqlite_master
        #         WHERE type IN ('table', 'view')
        #           AND name = ? COLLATE NOCASE
        #         """,
        #         (table_name,),
        #     ).fetchone()

        #     if table_row is None:
        #         raise ValueError(f"Table or view does not exist: {table_name}")

        #     actual_table_name = table_row[0]
        #     quoted_table = self._quote_identifier(actual_table_name)

        #     table_info = connection.execute(
        #         f"PRAGMA table_info({quoted_table})"
        #     ).fetchall()

        #     column_names = [column[1] for column in table_info]
        #     geometry_matches = [
        #         column
        #         for column in column_names
        #         if column.casefold() == geometry_column.casefold()
        #     ]

        #     if not geometry_matches:
        #         raise ValueError(
        #             f"Geometry column '{geometry_column}' was not found "
        #             f"in table '{actual_table_name}'. Available columns: "
        #             f"{column_names}"
        #         )

        #     if len(geometry_matches) > 1:
        #         raise ValueError(
        #             f"Multiple columns match geometry column "
        #             f"'{geometry_column}': {geometry_matches}"
        #         )

        #     actual_geometry_column = geometry_matches[0]
        #     attribute_columns = [
        #         column
        #         for column in column_names
        #         if column != actual_geometry_column
        #     ]

        #     wkt_alias = "_kalden_wkt_geometry"
        #     while wkt_alias in column_names:
        #         wkt_alias = f"_{wkt_alias}"

        #     select_expressions = [
        #         self._quote_identifier(column)
        #         for column in attribute_columns
        #     ]
        #     select_expressions.append(
        #         f"AsText({self._quote_identifier(actual_geometry_column)}) "
        #         f"AS {self._quote_identifier(wkt_alias)}"
        #     )

        #     query = f"""
        #         SELECT
        #             {", ".join(select_expressions)}
        #         FROM {quoted_table};
        #     """

        #     dataframe = pd.read_sql_query(query, connection)

        # dataframe["geometry"] = dataframe[wkt_alias].map(
        #     lambda value: loads(value)
        #     if isinstance(value, str) and value.strip()
        #     else None
        # )

        # return gpd.GeoDataFrame(
        #     dataframe.drop(columns=wkt_alias),
        #     geometry="geometry",
        #     crs=crs,
        # )

        # Safer connection: queries and SpatiaLite operate only on the temporary copy.
        # The original database is never passed to sqlite3.connect().
        with copied_sqlite_connection(self.db_path) as connection:
            connection.enable_load_extension(True)

            try:
                connection.execute('SELECT load_extension("mod_spatialite")')
            except sqlite3.Error as exc:
                raise RuntimeError(
                    "Could not load the 'mod_spatialite' SQLite extension."
                ) from exc
            finally:
                # Loading extensions should only be enabled for the shortest
                # possible period.
                connection.enable_load_extension(False)

            table_row = connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type IN ('table', 'view')
                AND name = ? COLLATE NOCASE
                """,
                (table_name,),
            ).fetchone()

            if table_row is None:
                raise ValueError(
                    f"Table or view does not exist: {table_name}"
                )

            actual_table_name = table_row[0]
            quoted_table = self._quote_identifier(actual_table_name)

            table_info = connection.execute(
                f"PRAGMA table_info({quoted_table})"
            ).fetchall()

            column_names = [column[1] for column in table_info]

            geometry_matches = [
                column
                for column in column_names
                if column.casefold() == geometry_column.casefold()
            ]

            if not geometry_matches:
                raise ValueError(
                    f"Geometry column '{geometry_column}' was not found "
                    f"in table '{actual_table_name}'. Available columns: "
                    f"{column_names}"
                )

            if len(geometry_matches) > 1:
                raise ValueError(
                    "Multiple columns match geometry column "
                    f"'{geometry_column}': {geometry_matches}"
                )

            actual_geometry_column = geometry_matches[0]

            attribute_columns = [
                column
                for column in column_names
                if column != actual_geometry_column
            ]

            wkt_alias = "_kalden_wkt_geometry"
            while wkt_alias in column_names:
                wkt_alias = f"_{wkt_alias}"

            select_expressions = [
                self._quote_identifier(column)
                for column in attribute_columns
            ]

            select_expressions.append(
                f"AsText({self._quote_identifier(actual_geometry_column)}) "
                f"AS {self._quote_identifier(wkt_alias)}"
            )

            query = f"""
                SELECT
                    {", ".join(select_expressions)}
                FROM {quoted_table};
            """

            dataframe = pd.read_sql_query(query, connection)

            dataframe["geometry"] = dataframe[wkt_alias].map(
                lambda value: (
                    loads(value)
                    if isinstance(value, str) and value.strip()
                    else None
                )
            )

            return gpd.GeoDataFrame(
                dataframe.drop(columns=wkt_alias),
                geometry="geometry",
                crs=crs,
            )


    @staticmethod
    def build_link_geometries_from_nodes(
        nodes_gdf: gpd.GeoDataFrame,
        links_df: pd.DataFrame,
        *,
        node_id_column: str = "MUID",
        from_node_column: str = "FromNodeID",
        to_node_column: str = "ToNodeID",
    ) -> gpd.GeoDataFrame:
        """
        Build straight link geometries from endpoint node geometries.

        Stored link geometry should generally be preferred because it may
        contain intermediate vertices. This helper is intended for tables
        without usable geometry or for deliberately simplified links.

        Args:
            nodes_gdf:
                GeoDataFrame containing node identifiers and point geometry.
            links_df:
                DataFrame containing upstream and downstream node identifiers.
            node_id_column:
                Node identifier column in ``nodes_gdf``.
            from_node_column:
                Upstream node identifier column in ``links_df``.
            to_node_column:
                Downstream node identifier column in ``links_df``.

        Returns:
            GeoDataFrame containing the link attributes and generated straight
            LineString geometries.
        """
        MPlusModel._require_active_geometry(nodes_gdf, "nodes_gdf")

        node_id_column = MPlusModel._resolve_column(
            nodes_gdf,
            node_id_column,
        )
        from_node_column = MPlusModel._resolve_column(
            links_df,
            from_node_column,
        )
        to_node_column = MPlusModel._resolve_column(
            links_df,
            to_node_column,
        )

        geometry_column = nodes_gdf.geometry.name
        duplicate_node_ids = (
            nodes_gdf.loc[
                nodes_gdf[node_id_column].duplicated(keep=False),
                node_id_column,
            ]
            .dropna()
            .unique()
            .tolist()
        )

        if duplicate_node_ids:
            raise ValueError(
                "Node identifiers must be unique. Duplicate IDs: "
                + ", ".join(map(str, duplicate_node_ids))
            )

        valid_node_geometries = nodes_gdf.geometry.dropna()
        invalid_geometry_types = sorted(
            set(valid_node_geometries.geom_type) - {"Point"}
        )

        if invalid_geometry_types:
            raise ValueError(
                "Node geometry must contain Points only. Found: "
                + ", ".join(invalid_geometry_types)
            )

        from_nodes = nodes_gdf[[node_id_column, geometry_column]].rename(
            columns={
                node_id_column: from_node_column,
                geometry_column: "_from_geometry",
            }
        )
        to_nodes = nodes_gdf[[node_id_column, geometry_column]].rename(
            columns={
                node_id_column: to_node_column,
                geometry_column: "_to_geometry",
            }
        )

        result = links_df.copy()

        # A pre-existing link geometry is intentionally replaced.
        if isinstance(result, gpd.GeoDataFrame):
            result = pd.DataFrame(result)

        if "geometry" in result.columns:
            result = result.drop(columns="geometry")

        result = result.merge(
            from_nodes,
            on=from_node_column,
            how="left",
            validate="many_to_one",
        )
        result = result.merge(
            to_nodes,
            on=to_node_column,
            how="left",
            validate="many_to_one",
        )

        missing_endpoints = (
            result["_from_geometry"].isna()
            | result["_to_geometry"].isna()
        )

        if missing_endpoints.any():
            missing_links = result.loc[
                missing_endpoints,
                [from_node_column, to_node_column],
            ]

            examples = missing_links.head(10).to_dict(orient="records")
            raise ValueError(
                f"{int(missing_endpoints.sum())} link(s) reference missing "
                f"node geometry. First examples: {examples}"
            )

        result["geometry"] = [
            LineString([from_geometry, to_geometry])
            for from_geometry, to_geometry in zip(
                result["_from_geometry"],
                result["_to_geometry"],
            )
        ]

        return gpd.GeoDataFrame(
            result.drop(columns=["_from_geometry", "_to_geometry"]),
            geometry="geometry",
            crs=nodes_gdf.crs,
        )

    @staticmethod
    def build_catchment_connection_geometry(
        row: pd.Series,
        catchment_geometry_column: str,
        node_geometry_column: str,
    ) -> LineString | None:
        """Build a line from a catchment centroid to its connected node."""
        catchment_geometry = row[catchment_geometry_column]
        node_geometry = row[node_geometry_column]

        if pd.isna(catchment_geometry) or pd.isna(node_geometry):
            return None

        if catchment_geometry.is_empty or node_geometry.is_empty:
            return None

        if node_geometry.geom_type != "Point":
            raise ValueError(
                "The node geometry must be a Point; received "
                f"{node_geometry.geom_type}."
            )

        return LineString([catchment_geometry.centroid, node_geometry])

    @staticmethod
    def validate_catchment_connections(
        catchments_gdf: pd.DataFrame,
        catchment_connections_gdf: pd.DataFrame,
        *,
        catchment_id_column: str = "muid",
        connection_id_column: str = "catchid",
    ) -> bool:
        """
        Validate that every catchment has exactly one connection.

        The validation checks for null identifiers, duplicate catchment IDs,
        multiple connections per catchment, missing connections, and references
        to unknown catchments.

        Returns:
            ``True`` when all checks pass.

        Raises:
            ValueError: If one or more checks fail.
        """
        catchment_id_column = MPlusModel._resolve_column(
            catchments_gdf,
            catchment_id_column,
        )
        connection_id_column = MPlusModel._resolve_column(
            catchment_connections_gdf,
            connection_id_column,
        )

        catchment_ids = catchments_gdf[catchment_id_column]
        connection_ids = catchment_connections_gdf[connection_id_column]
        errors: list[str] = []

        null_catchment_count = int(catchment_ids.isna().sum())
        null_connection_count = int(connection_ids.isna().sum())

        if null_catchment_count:
            errors.append(
                f"{null_catchment_count} catchment(s) have a null "
                f"'{catchment_id_column}'."
            )

        if null_connection_count:
            errors.append(
                f"{null_connection_count} connection(s) have a null "
                f"'{connection_id_column}'."
            )

        valid_catchment_ids = catchment_ids.dropna()
        valid_connection_ids = connection_ids.dropna()

        duplicate_catchment_ids = (
            valid_catchment_ids[
                valid_catchment_ids.duplicated(keep=False)
            ]
            .unique()
            .tolist()
        )

        if duplicate_catchment_ids:
            errors.append(
                "Duplicate catchment identifiers: "
                + ", ".join(map(str, duplicate_catchment_ids))
                + "."
            )

        connection_counts = valid_connection_ids.value_counts()
        multiple_connections = connection_counts[
            connection_counts > 1
        ].to_dict()

        if multiple_connections:
            details = ", ".join(
                f"{catchment_id} ({count})"
                for catchment_id, count in multiple_connections.items()
            )
            errors.append(
                f"Catchments with multiple connections: {details}."
            )

        catchment_id_set = set(valid_catchment_ids)
        connection_id_set = set(valid_connection_ids)

        missing_connections = sorted(
            catchment_id_set - connection_id_set,
            key=str,
        )
        if missing_connections:
            errors.append(
                "Catchments without a connection: "
                + ", ".join(map(str, missing_connections))
                + "."
            )

        unknown_catchments = sorted(
            connection_id_set - catchment_id_set,
            key=str,
        )
        if unknown_catchments:
            errors.append(
                "Connections referencing unknown catchments: "
                + ", ".join(map(str, unknown_catchments))
                + "."
            )

        if errors:
            raise ValueError(
                "Invalid catchment connections:\n- "
                + "\n- ".join(errors)
            )

        return True

    @staticmethod
    def upstream_analysis(
        catchment_connections_gdf: gpd.GeoDataFrame,
        links_gdf: gpd.GeoDataFrame,
        target_node_id: Any,
        plot: bool = False,
        *,
        connection_node_column: str = "NodeID",
        catchment_id_column: str = "muid",
        catchment_geometry_column: str = "geometry",
        node_geometry_column: str = "geometry_node",
        from_node_column: str = "FromNodeID",
        to_node_column: str = "ToNodeID",
        verbose: bool = True,
    ) -> dict[str, Any]:
        """
        Calculate the network and catchments upstream of one target node.

        Args:
            catchment_connections_gdf:
                GeoDataFrame containing catchment-to-node connections.
            links_gdf:
                GeoDataFrame containing directed network links.
            target_node_id:
                Target node identifier.
            plot:
                Whether to create a Matplotlib overview plot.
            connection_node_column:
                Connected node ID column in ``catchment_connections_gdf``.
            catchment_id_column:
                Catchment identifier used for plot colouring.
            catchment_geometry_column:
                Catchment polygon geometry column.
            node_geometry_column:
                Connected node point geometry column.
            from_node_column:
                Upstream endpoint column in ``links_gdf``.
            to_node_column:
                Downstream endpoint column in ``links_gdf``.
            verbose:
                Whether to print summary information.

        Returns:
            Dictionary containing upstream nodes, upstream links, upstream
            catchments, and total catchment area in hectares.
        """
        MPlusModel._require_active_geometry(
            catchment_connections_gdf,
            "catchment_connections_gdf",
        )
        MPlusModel._require_active_geometry(links_gdf, "links_gdf")

        connection_node_column = MPlusModel._resolve_column(
            catchment_connections_gdf,
            connection_node_column,
        )
        catchment_geometry_column = MPlusModel._resolve_column(
            catchment_connections_gdf,
            catchment_geometry_column,
        )
        if plot:
            catchment_id_column = MPlusModel._resolve_column(
                catchment_connections_gdf,
                catchment_id_column,
            )

        node_geometry_column = MPlusModel._resolve_column(
            catchment_connections_gdf,
            node_geometry_column,
        )
        from_node_column = MPlusModel._resolve_column(
            links_gdf,
            from_node_column,
        )
        to_node_column = MPlusModel._resolve_column(
            links_gdf,
            to_node_column,
        )

        graph = nx.DiGraph()
        graph.add_edges_from(
            links_gdf[[from_node_column, to_node_column]]
            .dropna()
            .itertuples(index=False, name=None)
        )
        graph.add_nodes_from(
            catchment_connections_gdf[connection_node_column].dropna()
        )

        if target_node_id not in graph:
            raise ValueError(
                f"Target node '{target_node_id}' is not present in the "
                "network or catchment connections."
            )

        upstream_nodes = sorted(
            nx.ancestors(graph, target_node_id),
            key=str,
        )
        contributing_nodes = set(upstream_nodes)
        contributing_nodes.add(target_node_id)

        upstream_catchments = catchment_connections_gdf.loc[
            catchment_connections_gdf[connection_node_column].isin(
                contributing_nodes
            )
        ].copy()
        upstream_catchments = upstream_catchments.set_geometry(
            catchment_geometry_column
        )

        MPlusModel._require_projected_crs(
            upstream_catchments,
            "catchment area",
        )

        valid_catchment_geometry = (
            upstream_catchments.geometry.notna()
            & ~upstream_catchments.geometry.is_empty
        )
        total_area_ha = (
            upstream_catchments.loc[
                valid_catchment_geometry,
                upstream_catchments.geometry.name,
            ].area.sum()
            / 10_000
        )

        upstream_link_mask = (
            links_gdf[from_node_column].isin(contributing_nodes)
            & links_gdf[to_node_column].isin(contributing_nodes)
        )
        upstream_links = links_gdf.loc[upstream_link_mask].copy()

        if verbose:
            print(
                f"Graph: {graph.number_of_nodes()} nodes, "
                f"{graph.number_of_edges()} edges"
            )
            print(
                f"Upstream nodes: {len(upstream_nodes)}, "
                f"contributing nodes: {len(contributing_nodes)}"
            )
            print(f"Total catchment area: {total_area_ha:.3f} ha")

        if plot:
            figure, axis = plt.subplots(figsize=(12, 8))

            if not upstream_catchments.empty:
                upstream_catchments.plot(
                    column=catchment_id_column,
                    legend=True,
                    cmap="tab20",
                    ax=axis,
                    alpha=0.7,
                    edgecolor="black",
                    linewidth=0.8,
                )

            if not upstream_links.empty:
                upstream_links.plot(
                    ax=axis,
                    color="blue",
                    linewidth=2,
                    alpha=0.6,
                    label="Links",
                )

            node_view = upstream_catchments.set_geometry(
                node_geometry_column
            )
            valid_node_geometry = (
                node_view.geometry.notna()
                & ~node_view.geometry.is_empty
            )
            node_view = node_view.loc[valid_node_geometry]

            if not node_view.empty:
                node_view.plot(
                    ax=axis,
                    color="yellow",
                    edgecolor="black",
                    markersize=50,
                    label="Connected nodes",
                )

                target_node_view = node_view.loc[
                    node_view[connection_node_column] == target_node_id
                ]
                if not target_node_view.empty:
                    target_node_view.plot(
                        ax=axis,
                        color="red",
                        markersize=200,
                        marker="*",
                        label=f"Target: {target_node_id}",
                    )

            axis.set_title(
                "Upstream catchments and network for "
                f"{target_node_id}\n"
                f"Area: {total_area_ha:.2f} ha; "
                f"catchments: {len(upstream_catchments)}",
                fontsize=14,
                fontweight="bold",
            )
            axis.set_axis_off()
            axis.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            figure.tight_layout()
            plt.show()

        return {
            "upstream_nodes": upstream_nodes,
            "upstream_links_gdf": upstream_links,
            "upstream_catchments_gdf": upstream_catchments,
            "total_area_ha": float(total_area_ha),
        }

    @staticmethod
    def batch_upstream_analysis(
        catchment_connections_gdf: gpd.GeoDataFrame,
        links_gdf: gpd.GeoDataFrame,
        nodes_gdf: gpd.GeoDataFrame,
        export_path: str | PathLike[str] | None = None,
        *,
        connection_node_column: str = "NodeID",
        catchment_geometry_column: str = "geometry_catchment",
        node_id_column: str = "MUID",
        from_node_column: str = "FromNodeID",
        to_node_column: str = "ToNodeID",
        show_progress: bool = True,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Run upstream network analysis for every node.

        Catchment area is calculated from the direct area connected to each
        contributing node. Link length is calculated only from links whose two
        endpoints both belong to the target node's upstream subnetwork.

        Args:
            catchment_connections_gdf:
                GeoDataFrame containing catchment-to-node connections.
            links_gdf:
                GeoDataFrame containing directed links.
            nodes_gdf:
                GeoDataFrame containing all target node identifiers.
            export_path:
                Optional Excel output path.
            connection_node_column:
                Connected node ID column in ``catchment_connections_gdf``.
            catchment_geometry_column:
                Catchment polygon geometry column.
            node_id_column:
                Node identifier column in ``nodes_gdf``.
            from_node_column:
                Upstream endpoint column in ``links_gdf``.
            to_node_column:
                Downstream endpoint column in ``links_gdf``.
            show_progress:
                Whether to display a tqdm progress bar.
            verbose:
                Whether to print a summary.

        Returns:
            DataFrame containing upstream node count, link length, catchment
            count, and catchment area for each node.
        """
        MPlusModel._require_active_geometry(
            catchment_connections_gdf,
            "catchment_connections_gdf",
        )
        MPlusModel._require_active_geometry(links_gdf, "links_gdf")
        MPlusModel._require_active_geometry(nodes_gdf, "nodes_gdf")

        connection_node_column = MPlusModel._resolve_column(
            catchment_connections_gdf,
            connection_node_column,
        )
        catchment_geometry_column = MPlusModel._resolve_column(
            catchment_connections_gdf,
            catchment_geometry_column,
        )
        node_id_column = MPlusModel._resolve_column(
            nodes_gdf,
            node_id_column,
        )
        from_node_column = MPlusModel._resolve_column(
            links_gdf,
            from_node_column,
        )
        to_node_column = MPlusModel._resolve_column(
            links_gdf,
            to_node_column,
        )

        catchment_view = catchment_connections_gdf.set_geometry(
            catchment_geometry_column
        ).copy()
        MPlusModel._require_projected_crs(
            catchment_view,
            "catchment area",
        )
        MPlusModel._require_projected_crs(links_gdf, "link length")

        graph = nx.DiGraph()
        graph.add_edges_from(
            links_gdf[[from_node_column, to_node_column]]
            .dropna()
            .itertuples(index=False, name=None)
        )
        graph.add_nodes_from(nodes_gdf[node_id_column].dropna())
        graph.add_nodes_from(
            catchment_view[connection_node_column].dropna()
        )

        valid_catchment_geometry = (
            catchment_view.geometry.notna()
            & ~catchment_view.geometry.is_empty
        )
        catchment_view = catchment_view.loc[
            valid_catchment_geometry
        ].copy()
        catchment_view["_kalden_area_ha"] = (
            catchment_view.geometry.area / 10_000
        )

        direct_area_by_node = catchment_view.groupby(
            connection_node_column,
            dropna=False,
        )["_kalden_area_ha"].sum()
        direct_count_by_node = catchment_view.groupby(
            connection_node_column,
            dropna=False,
        ).size()

        node_ids = nodes_gdf[node_id_column].dropna().drop_duplicates().tolist()
        iterator = tqdm(
            node_ids,
            desc="Upstream analysis",
            disable=not show_progress,
        )

        results: list[dict[str, Any]] = []

        for node_id in iterator:
            upstream_nodes = set(nx.ancestors(graph, node_id))
            contributing_nodes = upstream_nodes | {node_id}

            total_area_ha = sum(
                float(direct_area_by_node.get(upstream_node, 0.0))
                for upstream_node in contributing_nodes
            )
            catchment_count = sum(
                int(direct_count_by_node.get(upstream_node, 0))
                for upstream_node in contributing_nodes
            )

            upstream_link_mask = (
                links_gdf[from_node_column].isin(contributing_nodes)
                & links_gdf[to_node_column].isin(contributing_nodes)
            )
            pipe_length_m = links_gdf.loc[
                upstream_link_mask
            ].geometry.length.sum()

            results.append(
                {
                    "NodeID": node_id,
                    "n_upstream_nodes": len(upstream_nodes),
                    "n_upstream_catchments": catchment_count,
                    "upstream_pipe_length_m": float(pipe_length_m),
                    "upstream_catchment_area_ha": float(total_area_ha),
                }
            )

        results_df = pd.DataFrame(results)

        if not results_df.empty:
            results_df["n_upstream_nodes"] = results_df[
                "n_upstream_nodes"
            ].astype(int)
            results_df["n_upstream_catchments"] = results_df[
                "n_upstream_catchments"
            ].astype(int)

        if verbose:
            print(f"\nBatch analysis complete ({len(results_df)} nodes)")
            if not results_df.empty:
                print(results_df.describe())

        if export_path:
            output_path = Path(export_path).expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_excel(
                output_path,
                sheet_name="network analysis",
                index=False,
            )

            if verbose:
                print(f"Summary successfully exported to {output_path}")

        return results_df
