"""Lightweight helpers for reading and caching MIKE 1D ``.res1d`` files.

The module is intentionally small and dependency-light: it only requires
``pandas`` at import time. ``mikeio1d`` is imported lazily when a result file is
opened.

Design notes
------------
* Do not load every result series into memory.
* Open the ``.res1d`` file lazily and keep only the reader handle on the class.
* Read one time series at a time.
* Cache each requested time series on disk, next to the result file by default.
* Prefer parquet cache files when available, then fall back to pickle, then CSV.
* Classify special reach-backed objects such as ``Weir:<id>`` and ``Pump:<id>``
  as their own object types: ``weir`` and ``pump``.

Examples
--------
>>> from kalden.core.mike.res1d import Res1D
>>> res = Res1D("/path/to/result.res1d")
>>> res.available_quantities(object_type="node")
>>> df = res.read_series("node", "N1", "WaterLevel")
>>> weir_df = res.read_series("weir", "W1", "Discharge")
>>> res.warm_cache(quantities=["WaterLevel"], object_types=["node"])
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from os import PathLike as OsPathLike
from pathlib import Path
from typing import Any

import hashlib
import re
import shutil
import warnings

import pandas as pd

PathLike = str | OsPathLike[str]

__all__ = [
    "DEFAULT_OBJECT_TYPES",
    "DEFAULT_QUANTITY_CANDIDATES",
    "PathLike",
    "ObjectRef",
    "Res1D",
    "Res1DExplorer",
    "SeriesRef",
    "SPECIAL_REACH_PREFIXES",
    "default_cache_dir",
]


DEFAULT_QUANTITY_CANDIDATES: tuple[str, ...] = (
    "WaterLevel",
    "Discharge",
    "Flow",
    "Volume",
    "WaterDepth",
    "WaterVolume",
    "Velocity",
    "HeadLoss",
    "Inflow",
    "Outflow",
)

# MIKE 1D stores these as reaches, but exposes their IDs with prefixes such as
# "Weir:W1" or "Pump:P1". We expose them as dedicated object types while still
# resolving reads against ``res.reaches`` internally.
SPECIAL_REACH_PREFIXES: dict[str, str] = {
    "Weir": "weir",
    "Pump": "pump",
}

DEFAULT_OBJECT_TYPES: tuple[str, ...] = (
    "node",
    "reach",
    "weir",
    "pump",
)

_CACHE_SCHEMA_VERSION = "v2"


@dataclass(frozen=True, slots=True)
class ObjectRef:
    """Reference to one logical object in a ``.res1d`` file.

    ``object_type`` and ``object_id`` are the public, logical reference exposed
    by this module. ``source_object_type`` and ``source_object_id`` describe
    where the object is stored in the underlying MIKE result object.

    Example
    -------
    A MIKE reach with ID ``"Weir:W1"`` is exposed as::

        ObjectRef(
            object_type="weir",
            object_id="W1",
            source_object_type="reach",
            source_object_id="Weir:W1",
        )
    """

    object_type: str
    object_id: str
    source_object_type: str
    source_object_id: str


@dataclass(frozen=True, slots=True)
class SeriesRef:
    """Reference to one result time series in a ``.res1d`` file."""

    object_type: str
    object_id: str
    quantity: str


def default_cache_dir(res1d_path: PathLike) -> Path:
    """Return the default cache directory for a result file.

    The cache is placed in the same directory as the result file, not in the
    installed package and not necessarily in the current working directory.

    Examples
    --------
    ``/data/run_01.res1d`` -> ``/data/run_01_res1d_cache``
    """

    path = Path(res1d_path).expanduser()
    return path.parent / f"{path.stem}_res1d_cache"


def _safe_cache_token(value: object, *, max_length: int = 80) -> str:
    """Return a filesystem-safe token with a short hash to avoid collisions."""

    text = str(value)
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._") or "item"
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{slug[:max_length]}__{digest}"


def _canonical_special_prefix(object_type: str) -> str | None:
    for prefix, special_type in SPECIAL_REACH_PREFIXES.items():
        if special_type == object_type:
            return prefix
    return None


def _split_special_reach_id(object_id: object) -> tuple[str, str, str] | None:
    """Return ``(object_type, clean_id, canonical_raw_id)`` for prefixed IDs.

    The function is intentionally permissive about spacing around ``:`` because
    some result files/tools may display IDs as ``"Weir:W1"`` while others may
    display ``"Weir: W1"``.
    """

    text = str(object_id).strip()
    match = re.match(r"^([^:]+)\s*:\s*(.+)$", text)
    if match is None:
        return None

    raw_prefix = match.group(1).strip()
    clean_id = match.group(2).strip()
    if not clean_id:
        return None

    for canonical_prefix, object_type in SPECIAL_REACH_PREFIXES.items():
        if raw_prefix.lower() == canonical_prefix.lower():
            canonical_raw_id = f"{canonical_prefix}:{clean_id}"
            return object_type, clean_id, canonical_raw_id

    return None


def _object_ref_from_source(source_object_type: str, source_object_id: object) -> ObjectRef:
    """Return the public object reference for an underlying MIKE object."""

    source_type = _normalize_object_type(source_object_type)
    source_id = str(source_object_id)

    if source_type == "reach":
        special = _split_special_reach_id(source_id)
        if special is not None:
            object_type, object_id, _canonical_raw_id = special
            return ObjectRef(
                object_type=object_type,
                object_id=object_id,
                source_object_type="reach",
                source_object_id=source_id,
            )

    return ObjectRef(
        object_type=source_type,
        object_id=source_id,
        source_object_type=source_type,
        source_object_id=source_id,
    )


def _normalize_object_type(object_type: str) -> str:
    normalized = object_type.strip().lower()
    aliases = {
        "node": "node",
        "nodes": "node",
        "reach": "reach",
        "reaches": "reach",
        "branch": "reach",
        "branches": "reach",
        "link": "reach",
        "links": "reach",
        "weir": "weir",
        "weirs": "weir",
        "pump": "pump",
        "pumps": "pump",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        allowed = "'node', 'reach', 'weir', 'pump'"
        raise ValueError(f"object_type must be one of: {allowed}.") from exc


def _normalize_object_id_for_type(object_type: str, object_id: object) -> str:
    """Return the public ID used by this module for an object type."""

    normalized_type = _normalize_object_type(object_type)
    text = str(object_id)

    if normalized_type in SPECIAL_REACH_PREFIXES.values():
        special = _split_special_reach_id(text)
        if special is not None and special[0] == normalized_type:
            return special[1]

    return text


def _storage_id_candidates(object_type: str, object_id: object) -> list[str]:
    """Return likely underlying collection IDs for a public object reference."""

    normalized_type = _normalize_object_type(object_type)
    text = str(object_id)

    if normalized_type in {"node", "reach"}:
        return [text]

    prefix = _canonical_special_prefix(normalized_type)
    if prefix is None:
        return [text]

    visible_id = _normalize_object_id_for_type(normalized_type, text)
    candidates = [
        text,
        f"{prefix}:{visible_id}",
        f"{prefix}: {visible_id}",
    ]
    return list(dict.fromkeys(candidates))


def _collection_keys(collection: Any) -> list[Any]:
    """Return keys from a MIKE collection-like object."""

    try:
        return list(collection)
    except Exception:
        pass

    keys = getattr(collection, "keys", None)
    if callable(keys):
        try:
            return list(keys())
        except Exception:
            pass

    return []


def _safe_getattr(obj: Any, names: Sequence[str], default: Any = None) -> Any:
    """Return the first non-``None`` attribute found on an object."""

    for name in names:
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        if value is not None:
            return value
    return default


def _public_readable_quantities(obj: Any) -> list[str]:
    """Return public attributes that look like MIKE readable quantities."""

    quantities: list[str] = []
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            attr = getattr(obj, name)
        except Exception:
            continue
        if callable(attr):
            continue
        if hasattr(attr, "read"):
            quantities.append(name)
    return sorted(set(quantities))


def _normalize_timeseries(data: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """Return a clean DataFrame with a datetime index named ``time``."""

    if isinstance(data, pd.Series):
        frame = data.to_frame(name=data.name or "value")
    else:
        frame = data.copy()

    if frame.empty:
        frame.index.name = "time"
        return frame

    frame.index = pd.to_datetime(frame.index)
    frame.index.name = "time"
    frame.columns = [str(column) for column in frame.columns]
    return frame.sort_index()


class Res1D:
    """Read MIKE 1D ``.res1d`` files with lazy, per-series caching.

    Parameters
    ----------
    path : str or path-like
        Path to the ``.res1d`` result file.
    cache : bool, default True
        Enable on-disk cache for metadata and time series.
    cache_dir : str or path-like, optional
        Cache directory. If omitted, a sibling directory next to the result file
        is used: ``<result stem>_res1d_cache``.
    keep_in_memory : bool, default False
        Keep time series DataFrames in a small in-process dictionary after they
        have been read. Leave this disabled for large result files.
    default_quantities : sequence of str, optional
        Fallback quantity names used when ``mikeio1d`` does not expose readable
        quantity attributes through introspection.
    """

    def __init__(
        self,
        path: PathLike,
        *,
        cache: bool = True,
        cache_dir: PathLike | None = None,
        keep_in_memory: bool = False,
        default_quantities: Sequence[str] = DEFAULT_QUANTITY_CANDIDATES,
    ) -> None:
        self.path = Path(path).expanduser()
        self.cache = cache
        self.cache_dir = (
            Path(cache_dir).expanduser()
            if cache_dir is not None
            else default_cache_dir(self.path)
        )
        self.keep_in_memory = keep_in_memory
        self.default_quantities = tuple(default_quantities)

        self._res: Any | None = None
        self._memory_cache: dict[SeriesRef, pd.DataFrame] = {}

    @property
    def res1d_path(self) -> Path:
        """Backward-compatible alias for :attr:`path`."""

        return self.path

    def open(self) -> Any:
        """Open the result file lazily with ``mikeio1d.open``.

        ``mikeio1d`` is intentionally imported here instead of at module import
        time so the package remains importable in environments without MIKE IO.
        """

        if self._res is None:
            if self.path.suffix.lower() != ".res1d":
                raise ValueError(f"Expected a .res1d file, got: {self.path}")
            if not self.path.is_file():
                raise FileNotFoundError(f"res1d file not found: {self.path}")

            import mikeio1d

            self._res = mikeio1d.open(str(self.path))
        return self._res

    @property
    def res(self) -> Any:
        """Return the opened ``mikeio1d`` result object."""

        return self.open()

    def close(self) -> None:
        """Close the underlying result object when it supports ``close``."""

        if self._res is not None:
            close = getattr(self._res, "close", None)
            if callable(close):
                close()
        self._res = None
        self._memory_cache.clear()

    def __enter__(self) -> "Res1D":
        self.open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # noqa: ANN001
        self.close()

    def _cache_stem(self, name: str) -> Path:
        return self.cache_dir / _safe_cache_token(f"{_CACHE_SCHEMA_VERSION}_{name}")

    def _series_cache_stem(self, ref: SeriesRef) -> Path:
        return (
            self.cache_dir
            / "series"
            / _safe_cache_token(ref.object_type)
            / _safe_cache_token(ref.quantity)
            / _safe_cache_token(ref.object_id)
        )

    @staticmethod
    def _cache_candidates(stem: Path) -> list[tuple[str, Path]]:
        return [
            ("parquet", stem.with_suffix(".parquet")),
            ("pickle", stem.with_suffix(".pkl")),
            ("csv", stem.with_suffix(".csv")),
        ]

    def _read_dataframe_cache(self, stem: Path) -> pd.DataFrame | None:
        if not self.cache:
            return None

        for fmt, file_path in self._cache_candidates(stem):
            if not file_path.exists():
                continue
            if fmt == "parquet":
                try:
                    return pd.read_parquet(file_path)
                except Exception:
                    continue
            if fmt == "pickle":
                return pd.read_pickle(file_path)
            if fmt == "csv":
                return pd.read_csv(file_path, index_col=0, parse_dates=True)

        return None

    def _write_dataframe_cache(self, frame: pd.DataFrame, stem: Path) -> Path | None:
        if not self.cache:
            return None

        stem.parent.mkdir(parents=True, exist_ok=True)
        errors: list[str] = []

        parquet_path = stem.with_suffix(".parquet")
        try:
            frame.to_parquet(parquet_path)
            return parquet_path
        except Exception as exc:
            errors.append(f"parquet: {exc}")

        pickle_path = stem.with_suffix(".pkl")
        try:
            frame.to_pickle(pickle_path)
            return pickle_path
        except Exception as exc:
            errors.append(f"pickle: {exc}")

        csv_path = stem.with_suffix(".csv")
        try:
            frame.to_csv(csv_path)
            return csv_path
        except Exception as exc:
            errors.append(f"csv: {exc}")

        message = "Could not write cache file (" + "; ".join(errors) + ")"
        raise OSError(message)

    def _iter_node_items(self) -> Iterator[tuple[str, Any]]:
        nodes = self.res.nodes
        for key in _collection_keys(nodes):
            try:
                yield str(key), nodes[key]
            except Exception:
                continue

    def _iter_raw_reach_items(self) -> Iterator[tuple[str, Any]]:
        reaches = self.res.reaches
        for key in _collection_keys(reaches):
            try:
                item = reaches[key]
            except Exception:
                continue

            if isinstance(item, list):
                for index, sub_item in enumerate(item):
                    name = _safe_getattr(
                        sub_item,
                        ["name", "Name", "id", "ID"],
                        f"{key}__{index}",
                    )
                    yield str(name), sub_item
            else:
                name = _safe_getattr(item, ["name", "Name", "id", "ID"], str(key))
                yield str(name), item

    def _iter_reach_items(self) -> Iterator[tuple[str, Any]]:
        """Yield regular, non-special reaches only.

        Reach-backed weirs and pumps are exposed by :meth:`_iter_object_items`
        as ``object_type='weir'`` and ``object_type='pump'``.
        """

        for source_id, item in self._iter_raw_reach_items():
            ref = _object_ref_from_source("reach", source_id)
            if ref.object_type == "reach":
                yield ref.object_id, item

    def _iter_object_items(self, object_type: str) -> Iterator[tuple[ObjectRef, Any]]:
        normalized = _normalize_object_type(object_type)

        if normalized == "node":
            for object_id, item in self._iter_node_items():
                yield _object_ref_from_source("node", object_id), item
            return

        for source_id, item in self._iter_raw_reach_items():
            ref = _object_ref_from_source("reach", source_id)
            if ref.object_type == normalized:
                yield ref, item

    def _iter_items(self, object_type: str) -> Iterator[tuple[str, Any]]:
        """Backward-compatible iterator yielding public IDs and items."""

        for ref, item in self._iter_object_items(object_type):
            yield ref.object_id, item

    def _lookup_object(self, object_type: str, object_id: str) -> Any:
        normalized = _normalize_object_type(object_type)
        public_id = _normalize_object_id_for_type(normalized, object_id)

        if normalized == "node":
            collection = self.res.nodes
            try:
                return collection[public_id]
            except Exception:
                pass

            for current_id, item in self._iter_node_items():
                if current_id == public_id or current_id == str(object_id):
                    return item

            raise KeyError(f"node object not found: {object_id!r}")

        # Reaches, weirs, and pumps are all stored under res.reaches.
        reaches = self.res.reaches
        storage_candidates = _storage_id_candidates(normalized, object_id)

        for candidate in storage_candidates:
            try:
                item = reaches[candidate]
                if not isinstance(item, list):
                    return item
                if len(item) == 1:
                    return item[0]
            except Exception:
                pass

        storage_candidate_set = set(storage_candidates)
        input_id = str(object_id)

        for source_id, item in self._iter_raw_reach_items():
            if source_id in storage_candidate_set:
                return item

            ref = _object_ref_from_source("reach", source_id)
            if normalized == "reach":
                if ref.source_object_id == input_id or ref.object_id == input_id:
                    return item
                continue

            if ref.object_type == normalized and ref.object_id == public_id:
                return item

        raise KeyError(f"{normalized} object not found: {object_id!r}")

    def object_index(self, *, force_refresh: bool = False) -> pd.DataFrame:
        """Return known nodes, reaches, weirs, and pumps without reading values.

        Returns
        -------
        pandas.DataFrame
            Columns are ``object_type``, ``object_id``, ``source_object_type``,
            and ``source_object_id``. For normal nodes/reaches the public and
            source references are identical. For a special reach-backed object,
            for example ``Weir:W1``, the public reference is ``weir`` / ``W1``
            and the source reference is ``reach`` / ``Weir:W1``.
        """

        stem = self._cache_stem("object_index")
        if not force_refresh:
            cached = self._read_dataframe_cache(stem)
            if cached is not None:
                return cached

        rows: list[dict[str, str]] = []
        for object_type in DEFAULT_OBJECT_TYPES:
            for ref, _item in self._iter_object_items(object_type):
                rows.append(
                    {
                        "object_type": ref.object_type,
                        "object_id": ref.object_id,
                        "source_object_type": ref.source_object_type,
                        "source_object_id": ref.source_object_id,
                    }
                )

        index = pd.DataFrame(rows).drop_duplicates()
        if not index.empty:
            index = index.sort_values(["object_type", "object_id"]).reset_index(
                drop=True
            )

        self._write_dataframe_cache(index, stem)
        return index

    def available_object_types(self) -> list[str]:
        """Return object types present in the result file."""

        index = self.object_index()
        if index.empty:
            return []
        return sorted(index["object_type"].dropna().unique().tolist())

    def series_index(self, *, force_refresh: bool = False) -> pd.DataFrame:
        """Return available ``object_type``/``object_id``/``quantity`` rows.

        This reads only metadata-like attributes from the result object. It does
        not read every time series value from the ``.res1d`` file.

        The returned DataFrame also contains ``source_object_type`` and
        ``source_object_id`` so special objects can be traced back to their MIKE
        storage location.
        """

        stem = self._cache_stem("series_index")
        if not force_refresh:
            cached = self._read_dataframe_cache(stem)
            if cached is not None:
                return cached

        rows: list[dict[str, str]] = []
        for object_type in DEFAULT_OBJECT_TYPES:
            for ref, item in self._iter_object_items(object_type):
                quantities = _public_readable_quantities(item)
                if not quantities:
                    quantities = list(self.default_quantities)
                for quantity in quantities:
                    rows.append(
                        {
                            "object_type": ref.object_type,
                            "object_id": ref.object_id,
                            "quantity": quantity,
                            "source_object_type": ref.source_object_type,
                            "source_object_id": ref.source_object_id,
                        }
                    )

        index = pd.DataFrame(rows).drop_duplicates()
        if not index.empty:
            index = index.sort_values(
                ["object_type", "object_id", "quantity"]
            ).reset_index(drop=True)

        self._write_dataframe_cache(index, stem)
        return index

    def build_series_index(self, overwrite: bool = False) -> pd.DataFrame:
        """Backward-compatible wrapper around :meth:`series_index`."""

        return self.series_index(force_refresh=overwrite)

    def available_quantities(self, object_type: str | None = None) -> list[str]:
        """Return sorted quantity names, optionally filtered by object type."""

        index = self.series_index()
        if index.empty:
            return []
        if object_type is not None:
            normalized = _normalize_object_type(object_type)
            index = index[index["object_type"] == normalized]
        return sorted(index["quantity"].dropna().unique().tolist())

    def available_objects(
        self,
        quantity: str | None = None,
        object_type: str | None = None,
    ) -> list[str]:
        """Return sorted object identifiers, optionally filtered by quantity/type."""

        index = self.series_index()
        if index.empty:
            return []
        if object_type is not None:
            normalized = _normalize_object_type(object_type)
            index = index[index["object_type"] == normalized]
        if quantity is not None:
            index = index[index["quantity"] == quantity]
        return sorted(index["object_id"].dropna().unique().tolist())

    def cache_path(self, object_type: str, object_id: str, quantity: str) -> Path:
        """Return the cache stem used for a time series.

        The returned path has no suffix. Existing cache files may have
        ``.parquet``, ``.pkl``, or ``.csv`` suffixes.
        """

        normalized_type = _normalize_object_type(object_type)
        normalized_id = _normalize_object_id_for_type(normalized_type, object_id)
        ref = SeriesRef(
            object_type=normalized_type,
            object_id=normalized_id,
            quantity=str(quantity),
        )
        return self._series_cache_stem(ref)

    def read_series(
        self,
        object_type: str,
        object_id: str,
        quantity: str,
        *,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Read one result time series as a pandas DataFrame.

        The method checks, in order:
        1. optional in-memory cache,
        2. on-disk cache,
        3. the original ``.res1d`` file through ``mikeio1d``.

        ``object_type`` can be ``node``, ``reach``, ``weir``, or ``pump``. Weirs
        and pumps are resolved against the underlying ``res.reaches`` collection.
        These two calls are therefore equivalent when the result contains a
        source reach named ``"Weir:W1"``::

            res.read_series("weir", "W1", "Discharge")
            res.read_series("weir", "Weir:W1", "Discharge")
        """

        normalized_type = _normalize_object_type(object_type)
        normalized_id = _normalize_object_id_for_type(normalized_type, object_id)
        ref = SeriesRef(
            object_type=normalized_type,
            object_id=normalized_id,
            quantity=str(quantity),
        )

        if not force_refresh and self.keep_in_memory and ref in self._memory_cache:
            return self._memory_cache[ref].copy()

        stem = self._series_cache_stem(ref)
        if not force_refresh:
            cached = self._read_dataframe_cache(stem)
            if cached is not None:
                cached = _normalize_timeseries(cached)
                if self.keep_in_memory:
                    self._memory_cache[ref] = cached
                return cached.copy()

        obj = self._lookup_object(ref.object_type, ref.object_id)
        try:
            quantity_obj = getattr(obj, ref.quantity)
        except Exception as exc:
            raise AttributeError(
                f"{ref.object_type} {ref.object_id!r} has no quantity "
                f"{ref.quantity!r}."
            ) from exc

        read = getattr(quantity_obj, "read", None)
        if not callable(read):
            raise AttributeError(
                f"{ref.object_type} {ref.object_id!r} quantity "
                f"{ref.quantity!r} is not readable."
            )

        frame = _normalize_timeseries(read())
        self._write_dataframe_cache(frame, stem)
        if self.keep_in_memory:
            self._memory_cache[ref] = frame
        return frame.copy()

    def iter_series(
        self,
        refs: Iterable[SeriesRef],
        *,
        force_refresh: bool = False,
        errors: str = "raise",
    ) -> Iterator[tuple[SeriesRef, pd.DataFrame]]:
        """Yield several series one by one to avoid building a huge DataFrame."""

        for ref in refs:
            try:
                yield ref, self.read_series(
                    ref.object_type,
                    ref.object_id,
                    ref.quantity,
                    force_refresh=force_refresh,
                )
            except Exception as exc:
                if errors == "raise":
                    raise
                message = (
                    f"Could not read {ref.object_type}/"
                    f"{ref.object_id}/{ref.quantity}: {exc}"
                )
                if errors == "warn":
                    warnings.warn(message, stacklevel=2)
                elif errors != "ignore":
                    raise ValueError("errors must be one of: raise, warn, ignore") from exc

    def warm_cache(
        self,
        quantities: Iterable[str] | None = None,
        object_types: Iterable[str] = DEFAULT_OBJECT_TYPES,
        object_ids: Iterable[str] | None = None,
        max_items: int | None = None,
        force_refresh: bool = False,
        errors: str = "warn",
    ) -> list[SeriesRef]:
        """Populate the on-disk cache for selected series.

        This still reads one series at a time; it does not combine all results in
        memory. The returned list contains the series that were cached/read.
        """

        if errors not in {"raise", "warn", "ignore"}:
            raise ValueError("errors must be one of: raise, warn, ignore")

        index = self.series_index()
        if index.empty:
            return []

        normalized_types = {_normalize_object_type(item) for item in object_types}
        index = index[index["object_type"].isin(normalized_types)]

        if quantities is not None:
            index = index[index["quantity"].isin([str(item) for item in quantities])]

        if object_ids is not None:
            requested_ids = {str(item) for item in object_ids}
            index = index[index["object_id"].isin(requested_ids)]

        refs = [
            SeriesRef(row.object_type, row.object_id, row.quantity)
            for row in index.itertuples(index=False)
        ]
        if max_items is not None:
            refs = refs[:max_items]

        done: list[SeriesRef] = []
        for ref in refs:
            try:
                self.read_series(
                    ref.object_type,
                    ref.object_id,
                    ref.quantity,
                    force_refresh=force_refresh,
                )
                done.append(ref)
            except Exception as exc:
                message = (
                    f"Could not cache {ref.object_type}/"
                    f"{ref.object_id}/{ref.quantity}: {exc}"
                )
                if errors == "raise":
                    raise RuntimeError(message) from exc
                if errors == "warn":
                    warnings.warn(message, stacklevel=2)

        return done

    def clear_cache(self, *, include_memory: bool = True) -> None:
        """Delete this result file's cache directory."""

        if include_memory:
            self._memory_cache.clear()
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)

    def cache_size(self) -> int:
        """Return total cache size in bytes."""

        if not self.cache_dir.exists():
            return 0
        return sum(path.stat().st_size for path in self.cache_dir.rglob("*") if path.is_file())


# Backward-compatible alias for notebooks/scripts based on the exploratory file.
Res1DExplorer = Res1D
