"""
Utility functions for interacting with MIKE+ model files.

This module provides helper methods used across projects.

Author: DEAO
Created: 2026-01-15
"""

from datetime import datetime
import os
import sqlite3

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from shapely.geometry import LineString
from shapely.wkt import loads
from tqdm.notebook import tqdm


class MPlusModel:
    """Helper class for reading and analyzing MIKE+ model database content."""

    def __init__(self, db_path):
        """
        Initialize the MIKE+ model helper.

        Args:
            db_path: Path to the MIKE+ SQLite database file.
        """
        self.db_path = db_path

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
                Optional case-insensitive substring used to filter table names.
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
            ORDER BY name;
        """
    
        with sqlite3.connect(self.db_path) as con:
            rows = con.execute(query, object_types).fetchall()
    
        table_names = [row[0] for row in rows]
    
        if contains:
            search_value = contains.casefold()
            table_names = [
                name
                for name in table_names
                if search_value in name.casefold()
            ]
    
        if print_results:
            for name in table_names:
                print(name)
    
            print(f"\n{len(table_names)} table(s) found.")
    
        return table_names

    def fetch_table_attributes_geometry(
        self,
        table_name,
        geometry_column="Geometry",
        crs="EPSG:2056",
    ):
        """
        Fetch all attributes and geometry from a spatial MIKE+ database table.
    
        Args:
            table_name:
                Name of the spatial table to query.
            geometry_column:
                Name of the spatial geometry column.
            crs:
                Coordinate reference system assigned to the GeoDataFrame.
    
        Returns:
            A GeoDataFrame containing all table attributes and geometry,
            or None if the operation fails.
        """
    
        def quote_identifier(identifier):
            """Safely quote an SQLite table or column identifier."""
            return '"' + identifier.replace('"', '""') + '"'
    
        con = None
    
        try:
            con = sqlite3.connect(self.db_path)
            con.enable_load_extension(True)
            con.execute('SELECT load_extension("mod_spatialite")')
    
            table_exists = con.execute(
                """
                SELECT 1
                FROM sqlite_master
                WHERE type IN ('table', 'view')
                  AND name = ?
                """,
                (table_name,),
            ).fetchone()
    
            if table_exists is None:
                raise ValueError(f"Table does not exist: {table_name}")
    
            quoted_table = quote_identifier(table_name)
    
            table_info = con.execute(
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
                    f"in table '{table_name}'."
                )
    
            actual_geometry_column = geometry_matches[0]
    
            attribute_columns = [
                column
                for column in column_names
                if column.casefold() != actual_geometry_column.casefold()
            ]
    
            select_expressions = [
                quote_identifier(column)
                for column in attribute_columns
            ]
    
            select_expressions.append(
                f"AsText({quote_identifier(actual_geometry_column)}) "
                "AS wkt_geometry"
            )
    
            query = f"""
                SELECT
                    {", ".join(select_expressions)}
                FROM {quoted_table};
            """
    
            df = pd.read_sql_query(query, con)
    
            df["geometry"] = df["wkt_geometry"].apply(
                lambda value: loads(value)
                if value is not None and value != ""
                else None
            )
    
            return gpd.GeoDataFrame(
                df.drop(columns="wkt_geometry"),
                geometry="geometry",
                crs=crs,
            )
    
        except Exception as exc:
            print(
                f"Could not fetch attributes and geometry from "
                f"'{table_name}': {exc}"
            )
            return None
    
        finally:
            if con is not None:
                con.close()

    @staticmethod
    def build_link_geometries_from_nodes(
        nodes_gdf,
        links_df,
        *,
        node_id_column="MUID",
        from_node_column="FromNodeID",
        to_node_column="ToNodeID",
    ):
        """
        Build straight link geometries from their endpoint node geometries.
    
        This is intended for link tables that do not contain usable stored
        geometry. Existing link geometry from the database should generally
        be preferred because it may contain intermediate vertices.
        """
        required_node_columns = {
            node_id_column,
            nodes_gdf.geometry.name,
        }
        required_link_columns = {
            from_node_column,
            to_node_column,
        }
    
        missing_node_columns = required_node_columns - set(nodes_gdf.columns)
        missing_link_columns = required_link_columns - set(links_df.columns)
    
        if missing_node_columns:
            raise ValueError(
                "Missing node columns: "
                + ", ".join(sorted(missing_node_columns))
            )
    
        if missing_link_columns:
            raise ValueError(
                "Missing link columns: "
                + ", ".join(sorted(missing_link_columns))
            )
    
        node_geometries = nodes_gdf[
            [node_id_column, nodes_gdf.geometry.name]
        ].rename(
            columns={
                node_id_column: from_node_column,
                nodes_gdf.geometry.name: "_from_geometry",
            }
        )
    
        result = links_df.merge(
            node_geometries,
            on=from_node_column,
            how="left",
            validate="many_to_one",
        )
    
        node_geometries = nodes_gdf[
            [node_id_column, nodes_gdf.geometry.name]
        ].rename(
            columns={
                node_id_column: to_node_column,
                nodes_gdf.geometry.name: "_to_geometry",
            }
        )
    
        result = result.merge(
            node_geometries,
            on=to_node_column,
            how="left",
            validate="many_to_one",
        )
    
        missing_endpoints = (
            result["_from_geometry"].isna()
            | result["_to_geometry"].isna()
        )
    
        if missing_endpoints.any():
            invalid_links = result.loc[
                missing_endpoints,
                [from_node_column, to_node_column],
            ]
    
            raise ValueError(
                f"{len(invalid_links)} link(s) reference missing node geometry."
            )
    
        result["geometry"] = [
            LineString([from_geometry, to_geometry])
            for from_geometry, to_geometry in zip(
                result["_from_geometry"],
                result["_to_geometry"],
            )
        ]
    
        return gpd.GeoDataFrame(
            result.drop(
                columns=["_from_geometry", "_to_geometry"]
            ),
            geometry="geometry",
            crs=nodes_gdf.crs,
        )
    
    @staticmethod
    def build_catchment_connection_geometry(
        row,
        catchment_geometry_column,
        node_geometry_column,
    ):
        """
        Build a line from a catchment centroid to its connected node.
        """
        catchment_geometry = row[catchment_geometry_column]
        node_geometry = row[node_geometry_column]
    
        if catchment_geometry is None or node_geometry is None:
            return None
    
        if catchment_geometry.is_empty or node_geometry.is_empty:
            return None
    
        return LineString(
            [
                catchment_geometry.centroid,
                node_geometry,
            ]
        )

    @staticmethod
    def validate_catchment_connections(
        catchments_gdf,
        catchment_connections_gdf,
        *,
        catchment_id_column="muid",
        connection_id_column="catchid",
    ):
        """
        Validate that every catchment has exactly one connection.
    
        Checks that:
    
        1. No catchment has more than one connection.
        2. Every catchment has a connection.
        3. No connection references an unknown catchment.
        4. Catchment and connection identifiers are not null.
    
        Args:
            catchments_gdf:
                DataFrame or GeoDataFrame containing the catchments.
            catchment_connections_gdf:
                DataFrame or GeoDataFrame containing catchment connections.
            catchment_id_column:
                Catchment identifier column in ``catchments_gdf``.
            connection_id_column:
                Catchment identifier column in
                ``catchment_connections_gdf``.
    
        Returns:
            True when the connections are valid.
    
        Raises:
            ValueError:
                If the required columns are missing or validation fails.
        """
        if catchment_id_column not in catchments_gdf.columns:
            raise ValueError(
                f"Column '{catchment_id_column}' was not found in "
                "catchments_gdf."
            )
    
        if connection_id_column not in catchment_connections_gdf.columns:
            raise ValueError(
                f"Column '{connection_id_column}' was not found in "
                "catchment_connections_gdf."
            )
    
        catchment_ids = catchments_gdf[catchment_id_column]
        connection_ids = catchment_connections_gdf[connection_id_column]
    
        errors = []
    
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
                "Catchments with multiple connections: "
                f"{details}."
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
        catchments_connections_gdf,
        links_gdf,
        target_node_id,
        plot=False,
    ):
        """
        Compute upstream nodes and total catchment area draining into a target node.

        Assumes:
            - links_gdf has FromNodeID and ToNodeID columns.
            - catchments_connections_gdf has a NodeID column.
            - catchments_connections_gdf has geometry_catchment and geometry_node columns.

        Args:
            catchments_connections_gdf: GeoDataFrame linking catchments to nodes.
            links_gdf: GeoDataFrame containing network links.
            target_node_id: Target node identifier.
            plot: Whether to plot upstream catchments and links.

        Returns:
            Dictionary containing upstream nodes, upstream catchments, and total area in hectares.
        """
        graph = nx.DiGraph()

        for _, link in links_gdf.iterrows():
            graph.add_edge(link["FromNodeID"], link["ToNodeID"])

        graph.add_node(target_node_id)

        if target_node_id not in graph:
            return {"error": f"Node {target_node_id} not in graph"}

        upstream_nodes = list(nx.ancestors(graph, target_node_id))
        all_contributing_nodes = upstream_nodes + [target_node_id]

        print(
            f"Graph: {graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges"
        )
        print(
            f"Upstream nodes: {len(upstream_nodes)}, "
            f"Total contributing nodes: {len(all_contributing_nodes)}"
        )

        upstream_catch_gdf = catchments_connections_gdf[
            catchments_connections_gdf["NodeID"].isin(all_contributing_nodes)
        ].copy()

        upstream_catch_gdf = upstream_catch_gdf.set_geometry("geometry_catchment")
        total_area_ha = upstream_catch_gdf.geometry.area.sum() / 10_000

        print(f"total_area_ha : {total_area_ha}")

        if plot:
            upstream_links_gdf = links_gdf[
                links_gdf["ToNodeID"].isin(all_contributing_nodes)
            ]

            fig, ax = plt.subplots(figsize=(12, 8))

            upstream_catch_gdf.plot(
                column="muid",
                legend=True,
                cmap="tab20",
                ax=ax,
                alpha=0.7,
                edgecolor="black",
                linewidth=0.8,
            )

            upstream_links_gdf.plot(
                ax=ax,
                color="blue",
                linewidth=2,
                alpha=0.6,
                label="Links",
            )

            upstream_catch_gdf = upstream_catch_gdf.set_geometry("geometry_node")
            upstream_catch_gdf.plot(
                ax=ax,
                color="yellow",
                markersize=50,
                label="Nodes",
            )

            target_node_gdf = upstream_catch_gdf[
                upstream_catch_gdf["NodeID"] == target_node_id
            ]
            target_node_gdf.plot(
                ax=ax,
                color="red",
                markersize=200,
                marker="*",
                label=f"Target: {target_node_id}",
            )

            ax.set_title(
                f"Upstream Catchments + Network for Target Node {target_node_id}\n"
                f"Total Area: {total_area_ha:.2f} ha, "
                f"Catchments: {len(upstream_catch_gdf)}",
                fontsize=14,
                fontweight="bold",
            )
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            fig.tight_layout()
            plt.show()

        return {
            "upstream_nodes": upstream_nodes,
            "upstream_catchments_gdf": upstream_catch_gdf,
            "total_area_ha": total_area_ha,
        }

    @staticmethod
    def batch_upstream_analysis(
        catchments_connections_gdf,
        links_gdf,
        nodes_gdf,
        export_path="",
    ):
        """
        Run upstream catchment analysis for all nodes.

        The network graph is built once and reused for each node.

        Args:
            catchments_connections_gdf: GeoDataFrame linking catchments to nodes.
            links_gdf: GeoDataFrame containing FromNodeID, ToNodeID, and geometry columns.
            nodes_gdf: GeoDataFrame containing node MUID identifiers.
            export_path: Optional Excel output path for the summary table.

        Returns:
            DataFrame containing upstream node count, pipe length, and catchment area per node.
        """
        graph = nx.DiGraph()

        for _, link in links_gdf.iterrows():
            graph.add_edge(link["FromNodeID"], link["ToNodeID"])

        graph.add_nodes_from(nodes_gdf["MUID"])

        node_areas = {}

        for node_id in catchments_connections_gdf["NodeID"].unique():
            upstream_nodes = list(nx.ancestors(graph, node_id)) + [node_id]
            catch_gdf = catchments_connections_gdf[
                catchments_connections_gdf["NodeID"].isin(upstream_nodes)
            ].set_geometry("geometry_catchment")

            node_areas[node_id] = catch_gdf.geometry.area.sum() / 10_000

        def upstream_pipe_length(node_id):
            """
            Compute total upstream pipe length for a node.

            Args:
                node_id: Node identifier.

            Returns:
                Total upstream pipe length in meters.
            """
            upstream_nodes = list(nx.ancestors(graph, node_id)) + [node_id]
            upstream_links = links_gdf[
                links_gdf["FromNodeID"].isin(upstream_nodes)
                | links_gdf["ToNodeID"].isin(upstream_nodes)
            ]

            return upstream_links.geometry.length.sum()

        results = []

        for node_id in tqdm(nodes_gdf["MUID"]):
            upstream_nodes_count = len(list(nx.ancestors(graph, node_id)))
            total_area_ha = node_areas.get(node_id, 0)
            pipe_length_m = upstream_pipe_length(node_id)

            results.append(
                {
                    "NodeID": node_id,
                    "n_upstream_nodes": upstream_nodes_count,
                    "upstream_pipe_length_m": pipe_length_m,
                    "upstream_catchment_area_ha": total_area_ha,
                }
            )

        results_df = pd.DataFrame(results)
        results_df["n_upstream_nodes"] = results_df["n_upstream_nodes"].astype(int)

        print(f"\n🏭 BATCH ANALYSIS COMPLETE ({len(results_df)} nodes)")
        print(results_df.describe())

        if export_path != "":
            results_df.to_excel(
                export_path,
                sheet_name="network analysis",
                index=False,
            )
            print(f"Summary successfully exported to {export_path}")

        return results_df
