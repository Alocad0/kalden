from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .geopandas import GeoDataFrameViewer

__all__ = ["GeoDataFrameViewer"]


def __getattr__(name: str) -> Any:
    """Load optional geospatial helpers only when they are requested."""
    if name == "GeoDataFrameViewer":
        from .geopandas import GeoDataFrameViewer

        globals()[name] = GeoDataFrameViewer
        return GeoDataFrameViewer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
