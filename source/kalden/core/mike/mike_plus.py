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
    
    @staticmethod
    def fetch_links_geometry(
        nodes_gdf,
        links_gdf,
        node_suffix="_nodes",
        link_suffix="_links",
    ):
        """
        Build link geometries from MIKE+ node and link tables.

        Args:
            nodes_gdf: GeoDataFrame containing node geometries and MUID identifiers.
            links_gdf: GeoDataFrame or DataFrame containing FromNodeID and ToNodeID.
            node_suffix: Suffix applied to columns coming from nodes.
            link_suffix: Suffix applied to columns coming from links.

        Returns:
            GeoDataFrame with LineString geometries connecting upstream and downstream nodes.
        """
        links_gdf = links_gdf.drop_duplicates(
            subset=["FromNodeID", "ToNodeID"]
        ).copy()

        nodes_from = nodes_gdf.rename(
            columns={
                "geometry": f"from_geom{node_suffix}",
                "MUID": "FromNodeID",
            }
        )
        nodes_to = nodes_gdf.rename(
            columns={
                "geometry": f"to_geom{node_suffix}",
                "MUID": "ToNodeID",
            }
        )

        links_gdf = links_gdf.merge(
            nodes_from,
            on="FromNodeID",
            suffixes=(link_suffix, node_suffix),
        )
        links_gdf = links_gdf.merge(
            nodes_to,
            on="ToNodeID",
            suffixes=(link_suffix, node_suffix),
        )

        links_gdf["geometry"] = links_gdf.apply(
            lambda row: LineString(
                [
                    row[f"from_geom{node_suffix}"],
                    row[f"to_geom{node_suffix}"],
                ]
            ),
            axis=1,
        )

        lines_gdf = gpd.GeoDataFrame(
            links_gdf.drop(
                [f"from_geom{node_suffix}", f"to_geom{node_suffix}"],
                axis=1,
            ),
            geometry="geometry",
            crs=nodes_gdf.crs,
        )

        return lines_gdf

    def fetch_catchments_geometry(
        self,
        export=True,
        export_path="",
        overwrite=True,
    ):
        """
        Fetch catchment geometries from the MIKE+ database.

        Args:
            export: Whether to export the resulting GeoDataFrame to file.
            export_path: Output file path. If empty, a default shapefile path is used.
            overwrite: Whether to overwrite an existing export file.

        Returns:
            GeoDataFrame containing catchment geometries, or None if the operation fails.
        """
        con = None

        try:
            con = sqlite3.connect(self.db_path)
            con.enable_load_extension(True)
            con.execute('SELECT load_extension("mod_spatialite")')

            query = """
            SELECT MUID, AsText(Geometry) AS wkt_geometry
            FROM msm_Catchment;
            """

            df = pd.read_sql_query(query, con)
            df["geometry"] = df["wkt_geometry"].apply(loads)

            gdf = gpd.GeoDataFrame(
                df.drop("wkt_geometry", axis=1),
                geometry="geometry",
                crs="EPSG:2056",
            )

            con.close()
            con = None

            if export:
                if export_path == "":
                    export_path = os.path.join(
                        os.path.dirname(self.db_path),
                        "exported_shapefiles",
                        os.path.splitext(os.path.basename(self.db_path))[0] + ".shp",
                    )

                if os.path.exists(export_path) and not overwrite:
                    print(f"Could not export, file already exists: {export_path}")
                else:
                    os.makedirs(os.path.dirname(export_path), exist_ok=True)
                    gdf.to_file(export_path)
                    print(f"Export successful: {export_path}")

            self.catchments = gdf
            return gdf

        except Exception as exc:
            print(exc)
            return None

        finally:
            if con is not None:
                con.close()

    def fetch_table_geometry(
        self,
        table_name,
        export=True,
        export_path="",
        overwrite=True,
    ):
        """
        Fetch geometry from a spatial table in the MIKE+ database.

        Args:
            table_name: Name of the spatial table to query.
            export: Whether to export the resulting GeoDataFrame to file.
            export_path: Output file path. If empty, a default shapefile path is used.
            overwrite: Whether to overwrite an existing export file.

        Returns:
            GeoDataFrame containing the selected table geometry, or None if the operation fails.
        """
        con = None

        try:
            con = sqlite3.connect(self.db_path)
            con.enable_load_extension(True)
            con.execute('SELECT load_extension("mod_spatialite")')

            query = f"""
            SELECT MUID, AsText(Geometry) AS wkt_geometry
            FROM {table_name};
            """

            df = pd.read_sql_query(query, con)
            df["geometry"] = df["wkt_geometry"].apply(loads)

            gdf = gpd.GeoDataFrame(
                df.drop("wkt_geometry", axis=1),
                geometry="geometry",
                crs="EPSG:2056",
            )

            con.close()
            con = None

            if export:
                if export_path == "":
                    export_path = os.path.join(
                        os.path.dirname(self.db_path),
                        "exported_shapefiles",
                        os.path.splitext(os.path.basename(self.db_path))[0] + ".shp",
                    )

                if os.path.exists(export_path) and not overwrite:
                    print(f"Could not export, file already exists: {export_path}")
                else:
                    os.makedirs(os.path.dirname(export_path), exist_ok=True)
                    gdf.to_file(export_path)
                    print(f"Export successful: {export_path}")

            self.catchments = gdf
            return gdf

        except Exception as exc:
            print(exc)
            return None

        finally:
            if con is not None:
                con.close()

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
    def make_catchment_connection(row, col_geom_catch, col_geom_node):
        """
        Create a LineString connecting a catchment centroid to a node geometry.

        Args:
            row: DataFrame row containing catchment and node geometries.
            col_geom_catch: Column name containing the catchment geometry.
            col_geom_node: Column name containing the node geometry.

        Returns:
            LineString connecting the catchment centroid to the node geometry,
            or None if either geometry is missing.
        """
        if pd.isna(row[col_geom_catch]) or pd.isna(row[col_geom_node]):
            return None

        catch_centroid = row[col_geom_catch].centroid
        return LineString([catch_centroid, row[col_geom_node]])

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
