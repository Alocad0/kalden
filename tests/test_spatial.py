import geopandas as gpd
import pytest
from shapely.geometry import Point

from kalden.core.mike.mike_plus import MPlusModel
from kalden.core.spatial.dxf import DXFFile
from kalden.core.spatial.io import export_gdf


def test_dxf_empty_requested_type_has_geometry_column(tmp_path) -> None:
    ezdxf = pytest.importorskip("ezdxf")
    path = tmp_path / "empty.dxf"
    ezdxf.new().saveas(path)

    drawing = DXFFile(path, "EPSG:2056")
    drawing.extract_features(["LINE"])
    result = drawing.to_geodataframes()["LINE"]

    assert result.empty
    assert result.geometry.name == "geometry"


def test_gpkg_layer_overwrite_preserves_other_layers(tmp_path) -> None:
    path = tmp_path / "data.gpkg"
    original = gpd.GeoDataFrame(
        {"value": [1]},
        geometry=[Point(0, 0)],
        crs="EPSG:2056",
    )
    other = gpd.GeoDataFrame(
        {"value": [9]},
        geometry=[Point(9, 9)],
        crs="EPSG:2056",
    )
    replacement = gpd.GeoDataFrame(
        {"value": [2]},
        geometry=[Point(2, 2)],
        crs="EPSG:2056",
    )

    assert export_gdf(original, path, layer_name="target")
    assert export_gdf(other, path, layer_name="other")
    assert export_gdf(replacement, path, layer_name="target", overwrite=True)

    assert gpd.read_file(path, layer="target")["value"].tolist() == [2]
    assert gpd.read_file(path, layer="other")["value"].tolist() == [9]


def test_metric_calculations_reject_projected_crs_in_feet() -> None:
    frame = gpd.GeoDataFrame(
        geometry=[Point(0, 0)],
        crs="EPSG:2263",
    )

    with pytest.raises(ValueError, match="metre-based CRS"):
        MPlusModel._require_projected_crs(frame, "length")
