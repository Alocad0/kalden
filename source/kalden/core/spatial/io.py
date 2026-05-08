import os
import fiona
import geopandas as gpd
import sqlite3
from datetime import datetime
from pathlib import Path
from importlib.resources import files

def export_gdf(gdf, export_path, layer_name=None, export_file_type="gpkg", overwrite=False):
  """
  Safely export a GeoDataFrame to GeoPackage or Shapefile.

  Args:
      gdf: GeoDataFrame to export
      export_path: Full path to output file
          - GPKG example: 'data/lines.gpkg'
          - SHP example: 'data/lines.shp'
      layer_name: Layer name for GeoPackage export only
      export_file_type: 'gpkg' or 'shp' / 'shapefile'
      overwrite: Allow overwrite if exists

  Returns:
      bool: True if successful, False otherwise
  """
  def gpkg_layer_exists(gpkg_path, layer_name):
      if not os.path.exists(gpkg_path):
          return False
      return layer_name in fiona.listlayers(gpkg_path)

  def delete_gpkg_layer(gpkg_path, layer_name):
      """
      Delete a specific layer from a GeoPackage.

      Note:
          This follows the same approach as your original method.
          For more complete GeoPackage layer management, GDAL/OGR-based
          deletion can be more robust.
      """
      if not os.path.exists(gpkg_path):
          return False

      if layer_name not in fiona.listlayers(gpkg_path):
          return False

      import sqlite3
      con = sqlite3.connect(gpkg_path)
      try:
          con.execute(f'DROP TABLE IF EXISTS "{layer_name}"')
          con.commit()
      finally:
          con.close()

      print(f"Deleted layer: {layer_name}")
      return True

  def delete_shapefile(shp_path):
      """
      Delete all files belonging to a shapefile dataset.
      """
      base, _ = os.path.splitext(shp_path)
      sidecar_exts = [
          ".shp", ".shx", ".dbf", ".prj", ".cpg",
          ".sbn", ".sbx", ".qix", ".fix", ".xml"
      ]

      deleted = False
      for ext in sidecar_exts:
          file_path = base + ext
          if os.path.exists(file_path):
              os.remove(file_path)
              deleted = True

      if deleted:
          print(f"Deleted existing shapefile: {shp_path}")

      return deleted

  if not isinstance(gdf, gpd.GeoDataFrame):
      raise TypeError("gdf must be a GeoDataFrame")

  if gdf.empty:
      print("Warning: Empty GeoDataFrame; aborting export")
      return False

  export_file_type = export_file_type.lower()

  try:
      os.makedirs(os.path.dirname(export_path) or ".", exist_ok=True)

      if export_file_type in ["gpkg", "geopackage"]:
          if layer_name is None:
              raise ValueError("layer_name must be provided for GeoPackage export")

          gpkg_exists = os.path.exists(export_path)

          if not gpkg_exists:
              gdf.to_file(export_path, layer=layer_name, mode="w", driver="GPKG")

          else:
              layer_already_exists = gpkg_layer_exists(export_path, layer_name)

              if layer_already_exists and not overwrite:
                  print("Skipping - layer already exists and overwrite is set to False")
                  return False

              if layer_already_exists and overwrite:
                  delete_gpkg_layer(export_path, layer_name)

              gdf.to_file(export_path, layer=layer_name, mode="w", driver="GPKG")

          print(f"✓ Exported: {export_path} - {layer_name}")
          return True

      elif export_file_type in ["shp", "shapefile"]:
          if not export_path.lower().endswith(".shp"):
              raise ValueError("For shapefile export, export_path must end with '.shp'")

          shp_exists = os.path.exists(export_path)

          if shp_exists and not overwrite:
              print(f"Could not export, file already exists: {export_path}")
              return False

          if shp_exists and overwrite:
              delete_shapefile(export_path)

          gdf.to_file(export_path, driver="ESRI Shapefile")

          print(f"✓ Exported: {export_path}")
          return True

      else:
          raise ValueError("export_file_type must be 'gpkg', 'shp', or 'shapefile'")

  except Exception as e:
      print(f"✗ Export failed: {e}")
      return False

def insert_qml_style_into_gpkg(
    gpkg_path,
    layer_name,
    qml_path=None,
    builtin_style=None,
    style_name=None,
    description="",
    use_as_default=True,
    geometry_column=None,
    geometry_type=None,
    owner="",
):
      """
      Insert an existing QML style into a GeoPackage's QGIS layer_styles table.
      The QML style can be provided either as:
      - a file path via qml_path
      - a packaged built-in style via builtin_style

      Notes:
          - QGIS-specific, not a GeoPackage standard feature.
          - Stores QML only. styleSLD is left NULL.
          - Requires that the target layer already exists in the GeoPackage.

      Args:
          gpkg_path: path to .gpkg
          layer_name: target table/layer name in the GeoPackage
          qml_path: path to existing .qml file
          style_name: saved style name; defaults to layer_name
          builtin_style: name of builtin style
          description: optional description
          use_as_default: whether QGIS should use this as default style
          geometry_column: optional; autodetected from gpkg_geometry_columns if omitted
          geometry_type: optional; autodetected and normalized if omitted
          owner: optional metadata field

      Returns:
          bool
      """
    # Validate style source
    if qml_path is None and builtin_style is None:
        raise ValueError("Provide either qml_path or builtin_style.")

    if qml_path is not None and builtin_style is not None:
        raise ValueError("Provide only one of qml_path or builtin_style, not both.")

    # Read QML from external file
    if qml_path is not None:
        qml_path = Path(qml_path)

        if not qml_path.exists():
            raise FileNotFoundError(f"QML not found: {qml_path}")

        qml_text = qml_path.read_text(encoding="utf-8")

        if style_name is None:
            style_name = qml_path.stem

    # Read QML from package resources
    else:
        qml_filename = builtin_style

        if not qml_filename.endswith(".qml"):
            qml_filename = f"{qml_filename}.qml"

        qml_resource = files("kalden.styles").joinpath(qml_filename)

        if not qml_resource.is_file():
            raise FileNotFoundError(f"Built-in QML style not found: {qml_filename}")

        qml_text = qml_resource.read_text(encoding="utf-8")

        if style_name is None:
            style_name = Path(qml_filename).stem

    con = sqlite3.connect(gpkg_path)
    cur = con.cursor()

      try:
          # Confirm layer exists
          cur.execute(
              """
              SELECT COUNT(*)
              FROM gpkg_contents
              WHERE table_name = ?
              """,
              (layer_name,),
          )
          if cur.fetchone()[0] == 0:
              raise ValueError(f"Layer '{layer_name}' not found in GeoPackage")

          # Autodetect geometry column + geometry type
          if geometry_column is None or geometry_type is None:
              cur.execute(
                  """
                  SELECT column_name, geometry_type_name
                  FROM gpkg_geometry_columns
                  WHERE table_name = ?
                  """,
                  (layer_name,),
              )
              row = cur.fetchone()
              if row:
                  if geometry_column is None:
                      geometry_column = row[0]
                  if geometry_type is None:
                      gpkg_geom = (row[1] or "").upper()
                      if "POINT" in gpkg_geom:
                          geometry_type = "Point"
                      elif "LINE" in gpkg_geom:
                          geometry_type = "Line"
                      elif "POLYGON" in gpkg_geom:
                          geometry_type = "Polygon"
                      else:
                          geometry_type = None

          if geometry_column is None:
              geometry_column = "geom"

          # QGIS expects a layer_styles table for DB-backed styles
          cur.execute(
              """
              CREATE TABLE IF NOT EXISTS layer_styles (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  f_table_catalog TEXT,
                  f_table_schema TEXT,
                  f_table_name TEXT,
                  f_geometry_column TEXT,
                  styleName TEXT,
                  styleQML TEXT,
                  styleSLD TEXT,
                  useAsDefault INTEGER,
                  description TEXT,
                  owner TEXT,
                  ui TEXT,
                  update_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                  type TEXT
              )
              """
          )

          # Optional but useful indexes
          cur.execute(
              """
              CREATE INDEX IF NOT EXISTS idx_layer_styles_lookup
              ON layer_styles (f_table_name, f_geometry_column, styleName)
              """
          )

          # Only one default style per layer/geometry/type
          if use_as_default:
              cur.execute(
                  """
                  UPDATE layer_styles
                  SET useAsDefault = 0
                  WHERE f_table_name = ?
                  AND f_geometry_column = ?
                  AND (type = ? OR (? IS NULL AND type IS NULL))
                  """,
                  (layer_name, geometry_column, geometry_type, geometry_type),
              )

          # Remove existing style of same name for same layer
          cur.execute(
              """
              DELETE FROM layer_styles
              WHERE f_table_name = ?
              AND f_geometry_column = ?
              AND styleName = ?
              """,
              (layer_name, geometry_column, style_name),
          )

          # For SQLite-family providers, QGIS commonly stores datasource identity
          # fields in layer_styles. Using absolute gpkg path is the safest guess.
          f_table_catalog = os.path.abspath(gpkg_path)
          f_table_schema = ""

          cur.execute(
              """
              INSERT INTO layer_styles (
                  f_table_catalog,
                  f_table_schema,
                  f_table_name,
                  f_geometry_column,
                  styleName,
                  styleQML,
                  styleSLD,
                  useAsDefault,
                  description,
                  owner,
                  ui,
                  update_time,
                  type
              )
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
              """,
              (
                  f_table_catalog,
                  f_table_schema,
                  layer_name,
                  geometry_column,
                  style_name,
                  qml_text,
                  None,  # no SLD available
                  1 if use_as_default else 0,
                  description,
                  owner,
                  None,
                  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                  geometry_type,
              ),
          )

          con.commit()
          return True

      except Exception:
          con.rollback()
          raise

      finally:
          con.close()
