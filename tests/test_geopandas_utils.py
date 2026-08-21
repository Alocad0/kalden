from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from kalden.core.datascience.geopandas import GeoDataFrameViewer


def test_viewer_requires_a_geodataframe() -> None:
    with pytest.raises(TypeError, match="must be a geopandas.GeoDataFrame"):
        GeoDataFrameViewer(pd.DataFrame({"value": [1]}))


def test_prepare_geodataframe_filters_empty_geometry_and_reprojects() -> None:
    source = gpd.GeoDataFrame(
        {"name": ["valid", "missing", "empty"]},
        geometry=[Point(2_600_000, 1_200_000), None, Point()],
        crs="EPSG:2056",
    )
    viewer = GeoDataFrameViewer(source)

    result = viewer._prepare_geodataframe()

    assert result["name"].tolist() == ["valid"]
    assert result.crs.to_epsg() == 4326
    assert source.crs.to_epsg() == 2056
    assert len(source) == 3


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (
            gpd.GeoDataFrame(geometry=[], crs="EPSG:2056"),
            "empty GeoDataFrame",
        ),
        (
            gpd.GeoDataFrame(geometry=[Point(0, 0)]),
            "has no CRS",
        ),
        (
            gpd.GeoDataFrame(geometry=[None, Point()], crs="EPSG:2056"),
            "no non-empty geometries",
        ),
    ],
)
def test_prepare_geodataframe_rejects_unmappable_frames(frame, message) -> None:
    with pytest.raises(ValueError, match=message):
        GeoDataFrameViewer(frame)._prepare_geodataframe()


def test_viewer_temporary_html_path_is_reused_and_cleaned() -> None:
    viewer = GeoDataFrameViewer(
        gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:4326")
    )

    first = viewer._get_temp_html_path()
    second = viewer._get_temp_html_path()

    assert first == second
    assert first.is_file()
    assert viewer.html_path == first

    viewer.cleanup()
    assert not first.exists()
    assert viewer.html_path is None
