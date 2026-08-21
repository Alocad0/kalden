import geopandas as gpd
import pytest
from shapely.geometry import Point

from kalden.core.mike.mike_plus import MPlusModel
from kalden.core.spatial.dxf import DXFFile
from kalden.core.spatial.io import (
    export_gdf,
    insert_qml_style_into_gpkg,
    list_builtin_qml_styles,
    list_spatial_layers,
    read_spatial_file,
)


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


def test_export_gdf_rejects_non_spatial_or_empty_frames(tmp_path) -> None:
    with pytest.raises(TypeError, match="must be a GeoDataFrame"):
        export_gdf(
            {"value": [1]},
            tmp_path / "data.gpkg",
            layer_name="data",
        )

    empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:2056")
    assert not export_gdf(
        empty,
        tmp_path / "empty.gpkg",
        layer_name="empty",
    )
    assert not (tmp_path / "empty.gpkg").exists()


def test_read_spatial_file_requires_layer_for_multi_layer_gpkg(tmp_path) -> None:
    path = tmp_path / "multi.gpkg"
    first = gpd.GeoDataFrame(
        {"value": [1]},
        geometry=[Point(2_600_000, 1_200_000)],
        crs="EPSG:2056",
    )
    second = gpd.GeoDataFrame(
        {"value": [2]},
        geometry=[Point(2_600_100, 1_200_100)],
        crs="EPSG:2056",
    )
    assert export_gdf(first, path, layer_name="first")
    assert export_gdf(second, path, layer_name="second")
    assert list_spatial_layers(path) == ["first", "second"]

    with pytest.raises(ValueError, match="multiple layers"):
        read_spatial_file(path)

    result = read_spatial_file(path, layer="first", target_crs="EPSG:4326")
    assert result["value"].tolist() == [1]
    assert result.crs.to_epsg() == 4326


def test_builtin_qml_styles_are_packaged_and_extension_is_optional() -> None:
    names = list_builtin_qml_styles(print_styles=False)
    filenames = list_builtin_qml_styles(with_extension=True, print_styles=False)

    assert "mike_plus_nodes" in names
    assert "mike_plus_nodes.qml" in filenames
    assert names == sorted(names)
    assert all(not name.endswith(".qml") for name in names)


def test_insert_qml_style_validates_style_source_before_database_access(
    tmp_path,
) -> None:
    with pytest.raises(ValueError, match="Provide either"):
        insert_qml_style_into_gpkg(tmp_path / "missing.gpkg", "layer")

    with pytest.raises(ValueError, match="Provide only one"):
        insert_qml_style_into_gpkg(
            tmp_path / "missing.gpkg",
            "layer",
            qml_path=tmp_path / "style.qml",
            builtin_style="mike_plus_nodes",
        )

    with pytest.raises(FileNotFoundError, match="Built-in QML style not found"):
        insert_qml_style_into_gpkg(
            tmp_path / "missing.gpkg",
            "layer",
            builtin_style="does_not_exist",
        )
