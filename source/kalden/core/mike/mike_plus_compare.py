"""Scenario-aware comparison of two MIKE+ CS hydraulic topologies.

Install as ``kalden/core/mike/mike_plus_compare.py``.

V1 compares six resolved components by stable asset identifiers:
* node, link, weir, pump and catchment MUIDs;
* catchment connections by CatchID -> multiset of NodeID values.

Only connectivity and geometry are compared by default. To compare additional
scalar attributes (invert levels, diameter, etc.), supply ``attributes``.
No database file is modified. Database reading is delegated to MPlusModel.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import math
from numbers import Real
from pathlib import Path
from typing import Mapping, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd

from .mike_plus import MPlusModel, MPlusTopology, resolve_column


SPATIAL_LAYERS = ("nodes", "links", "weirs", "pumps", "catchments")
ALL_LAYERS = (*SPATIAL_LAYERS, "catchment_connections")
CHANGE_COLUMNS = ("layer", "asset_id", "change", "field", "before", "after")
SUMMARY_COLUMNS = ("layer", "before", "after", "added", "removed", "modified", "unchanged")
MAP_COLUMNS = ("asset_id", "change", "side", "fields", "geometry")


@dataclass
class MPlusComparison:
    """Summary, field-level changelog, and geospatial assets for GIS review."""

    before_scenario: str
    after_scenario: str
    summary: pd.DataFrame
    changes: pd.DataFrame
    spatial: dict[str, gpd.GeoDataFrame]

    def for_layer(self, layer: str) -> pd.DataFrame:
        """Return added/removed/modified records for one layer."""
        if layer not in ALL_LAYERS:
            raise ValueError(f"Unknown layer {layer!r}; expected one of {ALL_LAYERS}.")
        return self.changes.loc[self.changes["layer"] == layer].copy()

    def to_csv(self, directory: str | Path) -> None:
        """Export summary and field-level changelog to CSV."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.summary.to_csv(directory / "summary.csv", index=False)
        self.changes.to_csv(directory / "changes.csv", index=False)

    def to_geopackage(self, path: str | Path) -> list[str]:
        """Export changed spatial assets (both versions for modified assets).

        The GeoPackage contains only MUID, change type, version (before/after),
        changed field names and original geometry. Catchment connections are
        tabular-only in v1. Existing output files are intentionally refused.
        """
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing GeoPackage: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        written: list[str] = []
        for layer, gdf in self.spatial.items():
            non_null = gdf.loc[gdf.geometry.notna()].copy()
            if non_null.empty:
                continue
            # With distinct layer names, GeoPandas/GDAL adds successive layers
            # to the existing GeoPackage. No write occurs if all are empty.
            non_null.to_file(path, layer=layer, driver="GPKG")
            written.append(layer)
        return written


def _column_if_present(frame: pd.DataFrame, field: str) -> str | None:
    matches = [col for col in frame.columns if str(col).casefold() == field.casefold()]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous column {field!r}: {matches}")
    return matches[0] if matches else None


def _index_assets(frame: gpd.GeoDataFrame, layer: str) -> gpd.GeoDataFrame:
    id_col = resolve_column(frame, "MUID")
    ids = frame[id_col]
    if ids.isna().any() or (ids.astype(str).str.strip() == "").any():
        raise ValueError(f"{layer} contains empty MUID values.")
    if ids.astype(str).duplicated().any():
        dup = ids[ids.astype(str).duplicated()].iloc[0]
        raise ValueError(f"{layer} contains duplicate MUID {dup!r} after scenario resolution.")
    indexed = frame.copy()
    indexed.index = pd.Index(ids.astype(str), name="asset_id")
    return indexed


def _missing(value: object) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
        return bool(result) if isinstance(result, (bool, np.bool_)) else False
    except (TypeError, ValueError):
        return False


def _equal(left: object, right: object, *, atol: float) -> bool:
    if _missing(left) or _missing(right):
        return _missing(left) and _missing(right)
    if isinstance(left, Real) and isinstance(right, Real) and not (
        isinstance(left, bool) or isinstance(right, bool)
    ):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=atol)
    return bool(left == right)


def _geometry_equal(left: object, right: object, *, tolerance: float) -> bool:
    if _missing(left) or _missing(right):
        return _missing(left) and _missing(right)
    if left.is_empty and right.is_empty:
        return True
    if left.equals(right):
        return True
    return tolerance > 0 and left.hausdorff_distance(right) <= tolerance


def _geometry_signature(geometry: object) -> str | None:
    if _missing(geometry):
        return None
    return "sha256:" + hashlib.sha256(geometry.wkb).hexdigest()[:16]


def _connection_targets(frame: pd.DataFrame) -> dict[str, tuple[str | None, ...]]:
    catch_col = resolve_column(frame, "CatchID")
    node_col = resolve_column(frame, "NodeID")
    if frame[catch_col].isna().any() or (frame[catch_col].astype(str).str.strip() == "").any():
        raise ValueError("catchment_connections contains empty CatchID values.")
    groups: dict[str, list[str | None]] = defaultdict(list)
    for catch_id, node_id in frame[[catch_col, node_col]].itertuples(index=False, name=None):
        groups[str(catch_id)].append(None if _missing(node_id) else str(node_id))
    # Duplicates are retained, so removing a repeated connection is detected.
    return {
        catch_id: tuple(sorted(targets, key=lambda target: (target is None, str(target))))
        for catch_id, targets in groups.items()
    }


def _record(
    records: list[dict], layer: str, asset_id: str, change: str,
    field: str, before: object, after: object,
) -> None:
    records.append({
        "layer": layer, "asset_id": asset_id, "change": change,
        "field": field, "before": before, "after": after,
    })


def compare_topologies(
    before: MPlusTopology,
    after: MPlusTopology,
    *,
    attributes: Mapping[str, Sequence[str]] | None = None,
    geometry_tolerance: float = 0.0,
    numeric_tolerance: float = 1e-9,
) -> MPlusComparison:
    """Compare two scenario-resolved MIKE+ topologies.

    ``before`` is the reference and ``after`` is the candidate.
    Topology checks: entity existence (MUID), links/weirs/pumps endpoint
    references (where present), geometry and catchment-to-node connections.
    Additional scalar attributes are opt-in and compared case-insensitively
    by column name, with an absolute numeric tolerance.

    Geometries must share a CRS. ``geometry_tolerance`` is in CRS units
    (metres for EPSG:2056), and uses Hausdorff distance when nonzero.
    """
    if geometry_tolerance < 0 or numeric_tolerance < 0:
        raise ValueError("Tolerances must be nonnegative.")

    attributes = dict(attributes or {})
    unknown = set(attributes) - set(SPATIAL_LAYERS)
    if unknown:
        raise ValueError(f"Unknown/unsupported attribute layers: {sorted(unknown)}")

    summary_rows: list[dict] = []
    change_rows: list[dict] = []
    spatial: dict[str, gpd.GeoDataFrame] = {}

    for layer in SPATIAL_LAYERS:
        a = getattr(before, layer)
        b = getattr(after, layer)
        if not isinstance(a, gpd.GeoDataFrame) or not isinstance(b, gpd.GeoDataFrame):
            raise TypeError(f"{layer} must be a GeoDataFrame in both topologies.")
        if a.crs is None or b.crs is None or a.crs != b.crs:
            raise ValueError(f"{layer}: both versions must have the same defined CRS.")

        a_rows = _index_assets(a, layer)
        b_rows = _index_assets(b, layer)
        old_ids = set(a_rows.index)
        new_ids = set(b_rows.index)
        added = new_ids - old_ids
        removed = old_ids - new_ids
        shared = old_ids & new_ids

        fields: list[tuple[str, str, str]] = []
        for field in (("FromNodeID", "ToNodeID") if layer in ("links", "weirs", "pumps") else ()):
            a_col = _column_if_present(a_rows, field)
            b_col = _column_if_present(b_rows, field)
            if layer == "links" and shared and (a_col is None or b_col is None):
                raise ValueError(f"{layer}: required endpoint field {field!r} missing.")
            if shared and (a_col is None) != (b_col is None):
                raise ValueError(f"{layer}: endpoint field {field!r} present on one side only.")
            if a_col is not None and b_col is not None:
                fields.append((field, a_col, b_col))

        # Some MIKE+ structure tables have a single NodeID reference.
        if layer in ("weirs", "pumps"):
            a_col = _column_if_present(a_rows, "NodeID")
            b_col = _column_if_present(b_rows, "NodeID")
            if shared and (a_col is None) != (b_col is None):
                raise ValueError(f"{layer}: NodeID field present on one side only.")
            if a_col is not None and b_col is not None:
                fields.append(("NodeID", a_col, b_col))

        for field in attributes.get(layer, ()):
            if str(field).casefold() in {name.casefold() for name, _, _ in fields}:
                continue
            a_col = resolve_column(a_rows, field)
            b_col = resolve_column(b_rows, field)
            if a_col == a.geometry.name or b_col == b.geometry.name:
                raise ValueError("Geometry is compared automatically; do not list it in attributes.")
            if str(field).casefold() == "muid":
                raise ValueError("MUID is the matching key, not a comparable attribute.")
            fields.append((str(field), a_col, b_col))

        map_rows: list[dict] = []
        modified_count = 0

        def spatial_row(asset_id: str, change: str, side: str, changed_fields: str, row: pd.Series, *, geom_col: str) -> None:
            map_rows.append({
                "asset_id": asset_id, "change": change, "side": side,
                "fields": changed_fields, "geometry": row[geom_col],
            })

        for asset_id in sorted(added):
            _record(change_rows, layer, asset_id, "added", "__asset__", None, "present")
            spatial_row(asset_id, "added", "after", "__asset__", b_rows.loc[asset_id], geom_col=b.geometry.name)
        for asset_id in sorted(removed):
            _record(change_rows, layer, asset_id, "removed", "__asset__", "present", None)
            spatial_row(asset_id, "removed", "before", "__asset__", a_rows.loc[asset_id], geom_col=a.geometry.name)
        for asset_id in sorted(shared):
            a_row = a_rows.loc[asset_id]
            b_row = b_rows.loc[asset_id]
            changed: list[str] = []
            for field, a_col, b_col in fields:
                old_val, new_val = a_row[a_col], b_row[b_col]
                if not _equal(old_val, new_val, atol=numeric_tolerance):
                    changed.append(field)
                    _record(change_rows, layer, asset_id, "modified", field, old_val, new_val)
            ga, gb = a_row[a.geometry.name], b_row[b.geometry.name]
            if not _geometry_equal(ga, gb, tolerance=geometry_tolerance):
                changed.append("geometry")
                _record(change_rows, layer, asset_id, "modified", "geometry", _geometry_signature(ga), _geometry_signature(gb))
            if changed:
                modified_count += 1
                changed_fields = ", ".join(changed)
                spatial_row(asset_id, "modified", "before", changed_fields, a_row, geom_col=a.geometry.name)
                spatial_row(asset_id, "modified", "after", changed_fields, b_row, geom_col=b.geometry.name)

        spatial[layer] = gpd.GeoDataFrame(
            pd.DataFrame(map_rows, columns=MAP_COLUMNS),
            geometry="geometry", crs=a.crs,
        )
        summary_rows.append({
            "layer": layer, "before": len(old_ids), "after": len(new_ids),
            "added": len(added), "removed": len(removed),
            "modified": modified_count,
            "unchanged": len(shared) - modified_count,
        })

    # Catchment connections are grouped by CatchID rather than MUID.
    # This distinguishes reassignment A->B from two unrelated assets.
    old_targets = _connection_targets(before.catchment_connections)
    new_targets = _connection_targets(after.catchment_connections)
    old_ids = set(old_targets)
    new_ids = set(new_targets)
    added = new_ids - old_ids
    removed = old_ids - new_ids
    shared = old_ids & new_ids

    for catch_id in sorted(added):
        _record(change_rows, "catchment_connections", catch_id, "added", "NodeID", None, new_targets[catch_id])
    for catch_id in sorted(removed):
        _record(change_rows, "catchment_connections", catch_id, "removed", "NodeID", old_targets[catch_id], None)
    modified_count = 0
    for catch_id in sorted(shared):
        if old_targets[catch_id] != new_targets[catch_id]:
            modified_count += 1
            _record(change_rows, "catchment_connections", catch_id, "modified", "NodeID", old_targets[catch_id], new_targets[catch_id])
    summary_rows.append({
        "layer": "catchment_connections", "before": len(old_ids), "after": len(new_ids),
        "added": len(added), "removed": len(removed),
        "modified": modified_count, "unchanged": len(shared) - modified_count,
    })

    return MPlusComparison(
        before_scenario=before.scenario,
        after_scenario=after.scenario,
        summary=pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS),
        changes=pd.DataFrame(change_rows, columns=CHANGE_COLUMNS),
        spatial=spatial,
    )


def compare_models(
    before: MPlusModel | str | Path,
    after: MPlusModel | str | Path,
    *,
    before_scenario: str = "Base",
    after_scenario: str = "Base",
    crs: str = "EPSG:2056",
    validate: bool = True,
    attributes: Mapping[str, Sequence[str]] | None = None,
    geometry_tolerance: float = 0.0,
    numeric_tolerance: float = 1e-9,
) -> MPlusComparison:
    """Convenience wrapper: accept two MPlusModel objects or .sqlite/.mupp paths.

    The same model can be used twice with different scenario names.
    """
    before_model = before if isinstance(before, MPlusModel) else MPlusModel(before)
    after_model = after if isinstance(after, MPlusModel) else MPlusModel(after)
    top_before = before_model.fetch_topology(
        scenario=before_scenario, crs=crs, validate=validate,
    )
    top_after = after_model.fetch_topology(
        scenario=after_scenario, crs=crs, validate=validate,
    )
    return compare_topologies(
        top_before, top_after,
        attributes=attributes,
        geometry_tolerance=geometry_tolerance,
        numeric_tolerance=numeric_tolerance,
    )
