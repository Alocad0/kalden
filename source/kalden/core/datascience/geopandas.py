from __future__ import annotations

import atexit
import os
import tempfile
import warnings
import webbrowser
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import geopandas as gpd
from pandas.api.types import is_numeric_dtype

if TYPE_CHECKING:
    from folium import Map


FieldSelection = bool | str | int | Sequence[str]

_TEMP_MAP_FILES: set[Path] = set()


def _cleanup_temp_map_files() -> None:
    """Remove temporary map files when Python exits."""
    for path in tuple(_TEMP_MAP_FILES):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

        _TEMP_MAP_FILES.discard(path)


atexit.register(_cleanup_temp_map_files)


def _import_folium() -> tuple[Any, Any, Any]:
    """
    Import optional interactive-map dependencies.

    Keeping these imports local allows the rest of Kalden to work without
    Folium being installed.
    """
    try:
        import folium
        from folium.plugins import Fullscreen, MeasureControl
    except ImportError as exc:
        raise ImportError(
            "Interactive maps require Folium. Install the map dependencies "
            "with: pip install 'kalden[maps]'"
        ) from exc

    return folium, Fullscreen, MeasureControl


class GeoDataFrameViewer:
    """
    Interactive browser-based viewer for a GeoDataFrame.

    Maps are written to the operating system's temporary directory instead
    of the current workspace. Temporary files are removed when Python exits,
    or earlier by calling ``cleanup()``.
    """

    def __init__(self, gdf: gpd.GeoDataFrame) -> None:
        if not isinstance(gdf, gpd.GeoDataFrame):
            raise TypeError("gdf must be a geopandas.GeoDataFrame.")

        self.gdf = gdf
        self._map: Map | None = None
        self._html_path: Path | None = None

    @property
    def map(self) -> Map | None:
        """Most recently generated Folium map."""
        return self._map

    @property
    def html_path(self) -> Path | None:
        """Path of the temporary HTML map, if one has been created."""
        return self._html_path

    def create_map(
        self,
        column: str | None = None,
        tooltip: FieldSelection | None = None,
        popup: FieldSelection | None = None,
        categorical: bool | None = None,
        cmap: str = "viridis",
        tiles: str = "CartoDB Positron",
        layer_name: str = "Features",
        width: str | int = "100%",
        height: str | int = 650,
        fullscreen: bool = True,
        measure: bool = True,
        **explore_kwargs: Any,
    ) -> Map:
        """
        Create an interactive Folium map without opening the browser.

        Args:
            column:
                Optional attribute used to color the features.
            tooltip:
                Attributes shown when hovering. Defaults to the first five
                non-geometry columns.
            popup:
                Attributes shown when clicking. Defaults to the first ten
                non-geometry columns. Pass True to include every attribute.
            categorical:
                Whether ``column`` is categorical. When omitted, this is
                inferred from its dtype and unique-value count.
            cmap:
                Colormap used when ``column`` is supplied.
            tiles:
                Background map tiles.
            layer_name:
                Name of the GeoDataFrame layer.
            width:
                Map width.
            height:
                Map height.
            fullscreen:
                Add a fullscreen button.
            measure:
                Add a distance and area measurement tool.
            **explore_kwargs:
                Additional arguments passed to ``GeoDataFrame.explore()``.

        Returns:
            Generated Folium map.
        """
        folium, Fullscreen, MeasureControl = _import_folium()

        map_gdf = self._prepare_geodataframe()

        if column is not None and column not in map_gdf.columns:
            raise ValueError(
                f"Column '{column}' does not exist. "
                f"Available columns: {list(map_gdf.columns)}"
            )

        geometry_column = map_gdf.geometry.name
        attribute_columns = [
            name
            for name in map_gdf.columns
            if name != geometry_column
        ]

        if tooltip is None:
            tooltip = attribute_columns[:5]

        if popup is None:
            popup = attribute_columns[:10]

        if column is not None and categorical is None:
            values = map_gdf[column]

            categorical = (
                not is_numeric_dtype(values)
                or values.nunique(dropna=True) <= 12
            )

        options: dict[str, Any] = {
            "tiles": tiles,
            "tooltip": tooltip,
            "popup": popup,
            "highlight": True,
            "name": layer_name,
            "width": width,
            "height": height,
            "control_scale": True,
            "prefer_canvas": True,
            "style_kwds": {
                "weight": 2,
                "opacity": 0.9,
                "fillOpacity": 0.5,
            },
            "highlight_kwds": {
                "weight": 4,
                "fillOpacity": 0.75,
            },
            "marker_kwds": {
                "radius": 5,
                "fill": True,
                "fill_opacity": 0.85,
            },
            "map_kwds": {
                "scrollWheelZoom": True,
            },
        }

        if column is None:
            options["color"] = "#2878b5"
        else:
            options.update(
                {
                    "column": column,
                    "cmap": cmap,
                    "categorical": categorical,
                    "legend": True,
                    "legend_kwds": {
                        "caption": column,
                    },
                    "missing_kwds": {
                        "color": "#bdbdbd",
                        "label": "Missing",
                    },
                }
            )

        # Explicit arguments supplied by the caller take precedence.
        options.update(explore_kwargs)

        map_object = map_gdf.explore(**options)

        if fullscreen:
            Fullscreen(
                position="topright",
                title="Full screen",
                title_cancel="Exit full screen",
            ).add_to(map_object)

        if measure:
            MeasureControl(
                position="topright",
                primary_length_unit="meters",
                primary_area_unit="sqmeters",
            ).add_to(map_object)

        folium.LayerControl(
            position="topright",
            collapsed=True,
        ).add_to(map_object)

        self._map = map_object
        return map_object

    def show(self, **map_options: Any) -> Path:
        """
        Create the map, save it to a temporary HTML file, and open it.

        Unlike ``folium.Map.show_in_browser()``, this method does not keep
        the calling notebook cell running.

        Args:
            **map_options:
                Arguments passed to ``create_map()``.

        Returns:
            Path to the temporary HTML file.
        """
        map_object = self.create_map(**map_options)
        html_path = self._get_temp_html_path()

        map_object.save(str(html_path))

        opened = webbrowser.open_new_tab(html_path.as_uri())

        if not opened:
            warnings.warn(
                "The browser could not be opened automatically. "
                f"Open this file manually: {html_path}",
                stacklevel=2,
            )

        return html_path

    def cleanup(self) -> None:
        """Delete this viewer's temporary HTML file."""
        if self._html_path is None:
            return

        try:
            self._html_path.unlink(missing_ok=True)
        finally:
            _TEMP_MAP_FILES.discard(self._html_path)
            self._html_path = None

    def _prepare_geodataframe(self) -> gpd.GeoDataFrame:
        """Validate, clean, and project the GeoDataFrame for web mapping."""
        if self.gdf.empty:
            raise ValueError("Cannot display an empty GeoDataFrame.")

        if self.gdf.crs is None:
            raise ValueError(
                "The GeoDataFrame has no CRS. Set its CRS before mapping."
            )

        try:
            geometry = self.gdf.geometry
        except AttributeError as exc:
            raise ValueError(
                "The GeoDataFrame has no active geometry column."
            ) from exc

        map_gdf = self.gdf.loc[geometry.notna()].copy()
        map_gdf = map_gdf.loc[~map_gdf.geometry.is_empty]

        if map_gdf.empty:
            raise ValueError(
                "The GeoDataFrame contains no non-empty geometries."
            )

        # Web map GeoJSON coordinates should be longitude and latitude.
        return map_gdf.to_crs("EPSG:4326")

    def _get_temp_html_path(self) -> Path:
        """Create or reuse this viewer's temporary HTML file."""
        if self._html_path is not None:
            return self._html_path

        file_descriptor, filename = tempfile.mkstemp(
            prefix="kalden_map_",
            suffix=".html",
        )

        os.close(file_descriptor)

        self._html_path = Path(filename).resolve()
        _TEMP_MAP_FILES.add(self._html_path)

        return self._html_path
