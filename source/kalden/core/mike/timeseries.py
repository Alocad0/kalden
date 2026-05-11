"""Utilities for reading, validating, rewriting, and inspecting MIKE dfs0 time series.

This module wraps the current public ``mikeio`` dfs0 workflow:

* ``mikeio.open()`` for lightweight file/header access
* ``mikeio.read()`` for loading a dfs0 file into a ``Dataset``
* ``Dataset.to_dataframe(..., round_time=False)`` and ``mikeio.from_pandas(...)``
  when rebuilding a file with explicit timestamps and preserved item metadata
* ``Dataset.to_dfs()`` for writing

It also provides EUM catalogue utilities that do not require a dfs0 file:

* enumerate all available ``EUMType`` and ``EUMUnit`` values exposed by ``mikeio``
* search EUM types programmatically with ``mikeio.EUMType.search(...)``
* expand each EUM type to all valid units
* return the result as a pandas ``DataFrame`` by default

The implementation intentionally avoids the old script-style pattern of hard-coded
paths and top-level execution.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import os
from importlib.resources import as_file, files
from os import PathLike as OsPathLike
from pathlib import Path
from shutil import copy2
import tempfile

import mikeio
import pandas as pd

PathLike = str | OsPathLike[str]

__all__ = ["EUM", "Dfs0"]


class EUM:
    """Utilities for inspecting MIKE IO EUM types and units.

    This helper class is intentionally independent from any specific dfs format
    so it can be used with dfs0, dfs2, dfs3, or any other workflow relying on
    ``mikeio.EUMType`` and ``mikeio.EUMUnit``.
    """

    @staticmethod
    def iter_eum_types() -> list[mikeio.EUMType]:
        """Return all EUM types exposed by ``mikeio`` in deterministic order.

        The enumeration is built dynamically from ``mikeio.EUMType`` and does
        not depend on a dfs file being available.

        Returns
        -------
        list[mikeio.EUMType]
            All discovered EUM types sorted by display name.

        Examples
        --------
        >>> eum_types = EUM.iter_eum_types()
        >>> first = eum_types[0]
        >>> print(first)
        >>> print(first.name)
        >>> print(first.display_name)
        """
        eum_types: list[mikeio.EUMType] = []

        for name in dir(mikeio.EUMType):
            if name.startswith("_"):
                continue
            value = getattr(mikeio.EUMType, name)
            if isinstance(value, mikeio.EUMType):
                eum_types.append(value)

        return sorted(eum_types, key=lambda item: item.display_name.lower())

    @staticmethod
    def iter_eum_units() -> list[mikeio.EUMUnit]:
        """Return all EUM units exposed by ``mikeio`` in deterministic order.

        Returns
        -------
        list[mikeio.EUMUnit]
            All discovered EUM units sorted by display name.

        Examples
        --------
        >>> eum_units = EUM.iter_eum_units()
        >>> first = eum_units[0]
        >>> print(first)
        >>> print(first.name)
        >>> print(first.display_name)
        """
        eum_units: list[mikeio.EUMUnit] = []

        for name in dir(mikeio.EUMUnit):
            if name.startswith("_"):
                continue
            value = getattr(mikeio.EUMUnit, name)
            if isinstance(value, mikeio.EUMUnit):
                eum_units.append(value)

        return sorted(eum_units, key=lambda item: item.display_name.lower())

    @staticmethod
    def _build_eum_record(
        eum_type: mikeio.EUMType,
        eum_unit: mikeio.EUMUnit,
        *,
        include_objects: bool = False,
    ) -> dict[str, object]:
        """Build one serializable record for an EUM type/unit pair.

        Parameters
        ----------
        eum_type : mikeio.EUMType
            EUM type object.
        eum_unit : mikeio.EUMUnit
            EUM unit object valid for ``eum_type``.
        include_objects : bool, default False
            If ``True``, include the raw ``EUMType`` and ``EUMUnit`` objects in
            the returned record.

        Returns
        -------
        dict[str, object]
            Dictionary containing code-oriented names, display labels, and
            printable string representations.

        Notes
        -----
        The following fields expose the type/unit in different forms:

        * ``type_name`` / ``unit_name``:
          enum-style identifiers such as ``Temperature`` or ``degree_Kelvin``
        * ``type_display_name`` / ``unit_display_name``:
          user-friendly labels from MIKE IO
        * ``type_string`` / ``unit_string``:
          ``str(...)`` output for convenient printing or logging

        Examples
        --------
        >>> record = EUM._build_eum_record(
        ...     mikeio.EUMType.Temperature,
        ...     mikeio.EUMUnit.degree_Celsius,
        ... )
        >>> record["type_name"]
        'Temperature'
        >>> record["type_display_name"]
        'Temperature'
        >>> record["type_string"]
        >>> record["unit_name"]
        'degree_Celsius'
        >>> record["unit_display_name"]
        'degree Celsius'
        >>> record["unit_string"]
        """
        record: dict[str, object] = {
            "type_name": getattr(eum_type, "name", str(eum_type)),
            "type_display_name": eum_type.display_name,
            "type_string": str(eum_type),
            "unit_name": getattr(eum_unit, "name", str(eum_unit)),
            "unit_display_name": eum_unit.display_name,
            "unit_string": str(eum_unit),
        }

        if include_objects:
            record["type"] = eum_type
            record["unit"] = eum_unit

        return record

    @classmethod
    def catalog(
        cls,
        pattern: str | None = None,
        *,
        as_dataframe: bool = True,
        include_objects: bool = False,
    ) -> pd.DataFrame | list[dict[str, object]]:
        """Return the MIKE IO EUM type-to-unit catalogue.

        This utility does not require a dfs file. It can either:

        * search matching EUM types with ``mikeio.EUMType.search(pattern)``, or
        * enumerate every available EUM type exposed by ``mikeio``

        and then expand each type to all valid units.

        Parameters
        ----------
        pattern : str | None, optional
            Case-insensitive pattern used to filter EUM types. If omitted, the
            full EUM catalogue is returned.
        as_dataframe : bool, default True
            If ``True``, return a pandas ``DataFrame``. If ``False``, return a
            list of dictionaries.
        include_objects : bool, default False
            If ``True``, also include the raw ``EUMType`` and ``EUMUnit`` objects
            in each record.

        Returns
        -------
        pandas.DataFrame | list[dict[str, object]]
            Type/unit combinations, including both machine-oriented identifiers
            and human-readable string forms.

        Examples
        --------
        Return the full catalogue as a DataFrame:

        >>> df = EUM.catalog()

        Search only wind-related EUM types:

        >>> wind_df = EUM.catalog("wind")

        Return raw objects as dictionaries:

        >>> records = EUM.catalog("wind", as_dataframe=False, include_objects=True)
        >>> first = records[0]
        >>> print(first["type"])
        >>> print(first["type"].name)
        >>> print(first["type"].display_name)
        >>> print(first["unit"])
        >>> print(first["unit"].name)
        >>> print(first["unit"].display_name)
        """
        if pattern is None:
            eum_types = cls.iter_eum_types()
        else:
            normalized = pattern.strip()
            if not normalized:
                raise ValueError("pattern must be a non-empty string.")
            eum_types = sorted(
                mikeio.EUMType.search(normalized),
                key=lambda item: item.display_name.lower(),
            )

        rows: list[dict[str, object]] = []
        for eum_type in eum_types:
            for eum_unit in eum_type.units:
                rows.append(
                    cls._build_eum_record(
                        eum_type,
                        eum_unit,
                        include_objects=include_objects,
                    )
                )

        if as_dataframe:
            return pd.DataFrame(rows)

        return rows

    @classmethod
    def search(
        cls,
        pattern: str,
        *,
        as_dataframe: bool = True,
        include_objects: bool = False,
    ) -> pd.DataFrame | list[dict[str, object]]:
        """Search EUM types by name and expand each match to valid units.

        This is a convenience wrapper around :meth:`catalog` for the common use
        case corresponding to:

        ``mikeio.EUMType.search("wind")``

        Parameters
        ----------
        pattern : str
            Case-insensitive search pattern.
        as_dataframe : bool, default True
            If ``True``, return a pandas ``DataFrame``.
        include_objects : bool, default False
            If ``True``, include the raw ``EUMType`` and ``EUMUnit`` objects in
            each returned record.

        Returns
        -------
        pandas.DataFrame | list[dict[str, object]]
            Matching EUM type/unit combinations.

        Examples
        --------
        >>> df = EUM.search("wind")
        >>> print(df[["type_display_name", "unit_display_name"]])

        >>> records = EUM.search(
        ...     "wind",
        ...     as_dataframe=False,
        ...     include_objects=True,
        ... )
        >>> first = records[0]
        >>> print(first["type_string"])
        >>> print(first["unit_string"])
        """
        return cls.catalog(
            pattern=pattern,
            as_dataframe=as_dataframe,
            include_objects=include_objects,
        )


class Dfs0:
    """Convenience wrapper for common dfs0 file operations.

    Parameters
    ----------
    path : str | os.PathLike | None, optional
        Path to a dfs0 file. Methods that operate on a single file use this
        stored path when no explicit ``path`` argument is supplied.
    """

    def __init__(self, path: PathLike | None = None) -> None:
        self.path = None if path is None else self._as_path(path)

    @staticmethod
    def _as_path(path: PathLike) -> Path:
        """Return a normalized ``Path`` instance."""
        return Path(path).expanduser()

    @classmethod
    def _resolve_source(cls, path: PathLike | None, default: Path | None) -> Path:
        """Resolve and validate the source dfs0 path."""
        source = default if path is None else cls._as_path(path)
        if source is None:
            raise ValueError("A dfs0 file path must be provided.")
        if source.suffix.lower() != ".dfs0":
            raise ValueError(f"Expected a .dfs0 file, got: {source}")
        if not source.is_file():
            raise FileNotFoundError(f"dfs0 file not found: {source}")
        return source

    @staticmethod
    def _resolve_destination(
        source: Path,
        destination: PathLike | None,
        overwrite: bool,
    ) -> Path:
        """Resolve the output path and guard against accidental overwrites."""
        target = source if destination is None else Path(destination).expanduser()
        if target.suffix.lower() != ".dfs0":
            raise ValueError(f"Expected a .dfs0 file, got: {target}")
        if target.exists() and target != source and not overwrite:
            raise FileExistsError(f"Output file already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    @staticmethod
    def _validate_item_count(dataset, items: Sequence[object] | None) -> Sequence[object]:
        """Ensure replacement items match the number of data items."""
        if items is None:
            return list(dataset.items)
        if len(items) != dataset.n_items:
            raise ValueError(
                "Replacement items must match the number of dataset items "
                f"({dataset.n_items})."
            )
        return list(items)

    @classmethod
    def iter_files(cls, root: PathLike, *, recursive: bool = True) -> list[Path]:
        """Return all dfs0 files under ``root`` in deterministic order.

        Parameters
        ----------
        root : str | os.PathLike
            Either a single dfs0 file or a directory containing dfs0 files.
        recursive : bool, default True
            If ``True``, search subdirectories recursively.

        Returns
        -------
        list[pathlib.Path]
            Sorted list of dfs0 files.
        """
        root_path = cls._as_path(root)
        if root_path.is_file():
            if root_path.suffix.lower() != ".dfs0":
                raise ValueError(f"Expected a .dfs0 file, got: {root_path}")
            return [root_path]
        if not root_path.is_dir():
            raise FileNotFoundError(f"Directory not found: {root_path}")

        pattern = "**/*.dfs0" if recursive else "*.dfs0"
        return sorted(path for path in root_path.glob(pattern) if path.is_file())

    @staticmethod
    def _is_excluded(path: Path, exclude_substrings: Iterable[str]) -> bool:
        """Return ``True`` when a path should be skipped."""
        path_text = str(path).lower()
        return any(token in path_text for token in exclude_substrings)

    @classmethod
    def iter_eum_types(cls) -> list[mikeio.EUMType]:
        """Return all EUM types exposed by ``mikeio``.

        This wrapper is kept for backward compatibility. New code should call
        :meth:`EUM.iter_eum_types` directly.
        """
        return EUM.iter_eum_types()

    @classmethod
    def iter_eum_units(cls) -> list[mikeio.EUMUnit]:
        """Return all EUM units exposed by ``mikeio``.

        This wrapper is kept for backward compatibility. New code should call
        :meth:`EUM.iter_eum_units` directly.
        """
        return EUM.iter_eum_units()

    @classmethod
    def eum_catalog(
        cls,
        pattern: str | None = None,
        *,
        as_dataframe: bool = True,
        include_objects: bool = False,
    ) -> pd.DataFrame | list[dict[str, object]]:
        """Return the MIKE IO EUM type-to-unit catalogue.

        This wrapper is kept for backward compatibility. New code should call
        :meth:`EUM.catalog` directly.
        """
        return EUM.catalog(
            pattern=pattern,
            as_dataframe=as_dataframe,
            include_objects=include_objects,
        )

    @classmethod
    def search_eum_types(
        cls,
        pattern: str,
        *,
        as_dataframe: bool = True,
        include_objects: bool = False,
    ) -> pd.DataFrame | list[dict[str, object]]:
        """Search EUM types by name and expand each match to valid units.

        This wrapper is kept for backward compatibility. New code should call
        :meth:`EUM.search` directly.
        """
        return EUM.search(
            pattern=pattern,
            as_dataframe=as_dataframe,
            include_objects=include_objects,
        )

    def open(self, path: PathLike | None = None):
        """Open a dfs0 file with ``mikeio.open`` and return the header object."""
        source = self._resolve_source(path, self.path)
        return mikeio.open(source)

    def read(
        self,
        path: PathLike | None = None,
        *,
        items=None,
        time=None,
        keepdims: bool = False,
    ):
        """Read a dfs0 file into a ``mikeio.Dataset``.

        Parameters
        ----------
        path : str | os.PathLike | None, optional
            Path to the source dfs0 file. If omitted, the stored instance path is
            used.
        items : optional
            Item selection forwarded to ``mikeio.read``.
        time : optional
            Time selection forwarded to ``mikeio.read``.
        keepdims : bool, default False
            Forwarded to ``mikeio.read``.

        Returns
        -------
        mikeio.Dataset
            Loaded dataset.
        """
        source = self._resolve_source(path, self.path)
        return mikeio.read(source, items=items, time=time, keepdims=keepdims)

    def to_dataframe(
        self,
        path: PathLike | None = None,
        *,
        unit_in_name: bool = False,
        round_time: str | bool = "ms",
    ):
        """Read a dfs0 file and return a pandas ``DataFrame``.

        Parameters
        ----------
        path : str | os.PathLike | None, optional
            Source dfs0 file.
        unit_in_name : bool, default False
            Forwarded to ``Dataset.to_dataframe``.
        round_time : str | bool, default "ms"
            Forwarded to ``Dataset.to_dataframe``.

        Returns
        -------
        pandas.DataFrame
            DataFrame representation of the dfs0 dataset.
        """
        dataset = self.read(path)
        return dataset.to_dataframe(unit_in_name=unit_in_name, round_time=round_time)

    def duplicate_timestamps(self, path: PathLike | None = None):
        """Return duplicated timestamps from a dfs0 file.

        The result is a ``pandas.DatetimeIndex`` containing every duplicated
        timestamp occurrence (``keep=False``).

        Parameters
        ----------
        path : str | os.PathLike | None, optional
            Source dfs0 file.

        Returns
        -------
        pandas.DatetimeIndex
            Duplicate timestamps.
        """
        dataset = self.read(path)
        return dataset.time[dataset.time.duplicated(keep=False)]

    def validate_timestamps(
        self,
        path: PathLike | None = None,
        *,
        require_sorted: bool = True,
        require_unique: bool = True,
    ):
        """Validate timestamp ordering and uniqueness.

        Parameters
        ----------
        path : str | os.PathLike | None, optional
            Source dfs0 file.
        require_sorted : bool, default True
            Require timestamps to be monotonically increasing.
        require_unique : bool, default True
            Require timestamps to be unique.

        Returns
        -------
        mikeio.Dataset
            The loaded dataset, so validation and downstream work can share the
            same read operation.

        Raises
        ------
        ValueError
            If the file has no time steps, unsorted timestamps, or duplicate
            timestamps according to the selected checks.
        """
        source = self._resolve_source(path, self.path)
        dataset = self.read(source)

        if dataset.n_timesteps == 0:
            raise ValueError(f"dfs0 file has no time steps: {source}")
        if require_sorted and not dataset.time.is_monotonic_increasing:
            raise ValueError(f"Timestamps are not sorted in ascending order: {source}")
        if require_unique and dataset.time.has_duplicates:
            duplicates = dataset.time[dataset.time.duplicated(keep=False)]
            raise ValueError(
                f"Duplicate timestamps found in {source}: {duplicates.unique().tolist()}"
            )

        return dataset

    def rewrite(
        self,
        destination: PathLike | None = None,
        *,
        overwrite: bool = False,
        items: Sequence[object] | None = None,
        title: str | None = None,
        validate_timestamps: bool = True,
        **kwargs,
    ) -> Path:
        """Rewrite a dfs0 file using current public ``mikeio`` APIs.

        The file is read to a ``Dataset``, converted to a ``DataFrame`` without
        timestamp rounding, recreated with ``mikeio.from_pandas(...)`` to keep
        item metadata, and written back with ``Dataset.to_dfs()``.

        Parameters
        ----------
        destination : str | os.PathLike | None, optional
            Output path. If omitted, the source file is overwritten.
        overwrite : bool, default False
            Allow overwriting an existing output file when ``destination`` is a
            different path than the source.
        items : sequence, optional
            Replacement ``mikeio.ItemInfo`` sequence. If omitted, the original
            item metadata is preserved.
        title : str | None, optional
            Optional dfs title. Defaults to the source filename stem.
        validate_timestamps : bool, default True
            Validate timestamp ordering and uniqueness before writing.
        **kwargs
            Additional keyword arguments forwarded to ``Dataset.to_dfs()``.

        Returns
        -------
        pathlib.Path
            Path to the written dfs0 file.
        """
        source = self._resolve_source(None, self.path)
        target = self._resolve_destination(source, destination, overwrite)
        dataset = (
            self.validate_timestamps(source) if validate_timestamps else self.read(source)
        )

        output_items = self._validate_item_count(dataset, items)
        dataframe = dataset.to_dataframe(unit_in_name=False, round_time=False)
        rebuilt = mikeio.from_pandas(dataframe, items=output_items)
        rebuilt.to_dfs(target, title=source.stem if title is None else title, **kwargs)
        return target

    def convert_to_nonequidistant(
        self,
        destination: PathLike | None = None,
        *,
        overwrite: bool = False,
        items: Sequence[object] | None = None,
        title: str | None = None,
        validate_timestamps: bool = True,
        require_non_equidistant: bool = True,
        **kwargs,
    ) -> Path:
        """Rewrite a dfs0 file and verify that the output is non-equidistant.

        This method uses the same public ``mikeio`` dataset/DataFrame workflow as
        :meth:`rewrite`. After writing, it re-reads the output and can assert that
        the resulting time axis is non-equidistant.

        Parameters
        ----------
        destination : str | os.PathLike | None, optional
            Output path. If omitted, the source file is replaced in place.
        overwrite : bool, default False
            Allow overwriting an existing output file when ``destination`` differs
            from the source.
        items : sequence, optional
            Replacement ``mikeio.ItemInfo`` sequence.
        title : str | None, optional
            Optional dfs title.
        validate_timestamps : bool, default True
            Validate timestamp ordering and uniqueness before writing.
        require_non_equidistant : bool, default True
            If ``True``, raise an error when the rewritten file still has an
            equidistant time axis.
        **kwargs
            Additional keyword arguments forwarded to :meth:`rewrite`.

        Returns
        -------
        pathlib.Path
            Path to the rewritten dfs0 file.

        Raises
        ------
        RuntimeError
            If ``require_non_equidistant`` is ``True`` and the written file still
            has an equidistant time axis.
        """
        source = self._resolve_source(None, self.path)
        target = self._resolve_destination(source, destination, overwrite)
        target_for_write = target
        temp_target: Path | None = None

        if require_non_equidistant and target == source:
            handle, temp_name = tempfile.mkstemp(
                prefix=f"{source.stem}_",
                suffix=".dfs0",
                dir=source.parent,
            )
            os.close(handle)
            Path(temp_name).unlink(missing_ok=True)
            temp_target = Path(temp_name)

        if temp_target is not None:
            target_for_write = temp_target

        try:
            rewritten_target = self.rewrite(
                destination=target_for_write,
                overwrite=True,
                items=items,
                title=title,
                validate_timestamps=validate_timestamps,
                **kwargs,
            )

            if require_non_equidistant:
                written = mikeio.read(rewritten_target)
                if written.is_equidistant:
                    raise RuntimeError(
                        "The rewritten file is still equidistant. The current public "
                        "mikeio write path preserved the regular time axis."
                    )

            if temp_target is not None:
                temp_target.replace(source)
                return source

            return rewritten_target
        finally:
            if temp_target is not None and temp_target.exists():
                temp_target.unlink(missing_ok=True)

    @classmethod
    def batch_convert_to_nonequidistant(
        cls,
        root: PathLike,
        *,
        recursive: bool = True,
        overwrite: bool = False,
        exclude_substrings: Iterable[str] | None = None,
        require_non_equidistant: bool = True,
        **kwargs,
    ) -> list[Path]:
        """Convert every eligible dfs0 file below ``root``.

        Parameters
        ----------
        root : str | os.PathLike
            A dfs0 file or a directory containing dfs0 files.
        recursive : bool, default True
            Search subdirectories when ``root`` is a directory.
        overwrite : bool, default False
            Allow overwriting destination files when ``destination`` is supplied
            via ``kwargs`` for individual calls.
        exclude_substrings : iterable of str, optional
            Case-insensitive substrings used to skip matching paths.
        require_non_equidistant : bool, default True
            Propagate the non-equidistant verification check.
        **kwargs
            Additional keyword arguments forwarded to
            :meth:`convert_to_nonequidistant`.

        Returns
        -------
        list[pathlib.Path]
            Paths to converted files.
        """
        exclusions = tuple(token.lower() for token in (exclude_substrings or ()))
        converted: list[Path] = []

        for path in cls.iter_files(root, recursive=recursive):
            if exclusions and cls._is_excluded(path, exclusions):
                continue
            reader = cls(path)
            converted.append(
                reader.convert_to_nonequidistant(
                    overwrite=overwrite,
                    require_non_equidistant=require_non_equidistant,
                    **kwargs,
                )
            )

        return converted

    @classmethod
    def scan_duplicate_timestamps(
        cls,
        root: PathLike,
        *,
        recursive: bool = True,
        exclude_substrings: Iterable[str] | None = None,
    ) -> dict[Path, list]:
        """Scan one file or a directory tree for duplicate timestamps.

        Parameters
        ----------
        root : str | os.PathLike
            A dfs0 file or a directory containing dfs0 files.
        recursive : bool, default True
            Search subdirectories when ``root`` is a directory.
        exclude_substrings : iterable of str, optional
            Case-insensitive substrings used to skip matching paths.

        Returns
        -------
        dict[pathlib.Path, list]
            Mapping from dfs0 file path to duplicate timestamps.
        """
        exclusions = tuple(token.lower() for token in (exclude_substrings or ()))
        findings: dict[Path, list] = {}

        for path in cls.iter_files(root, recursive=recursive):
            if exclusions and cls._is_excluded(path, exclusions):
                continue
            duplicates = cls(path).duplicate_timestamps()
            if len(duplicates) > 0:
                findings[path] = duplicates.tolist()

        return findings

def download_template(template_path, target_path):
    """Copy a template file from ``kalden/templates`` to a target location.

    Parameters
    ----------
    template_path : str or pathlib.Path
        Path to the template file relative to ``kalden/templates``.

        Example
        -------
        ``"file.xlsx"``
        ``"method/file.xlsx"``

    target_path : str or pathlib.Path
        Destination path. If this is an existing directory, the template keeps
        its original filename. Otherwise, it is copied to the given file path.

    Returns
    -------
    pathlib.Path
        Path to the copied template file.
    """
    template_path = Path(template_path)
    target_path = Path(target_path)

    if template_path.is_absolute():
        raise ValueError(
            "template_path must be relative to 'kalden/templates', "
            "not an absolute path."
        )

    template_resource = files("kalden") / "templates" / template_path.as_posix()

    if not template_resource.is_file():
        raise FileNotFoundError(
            f"Template file not found in 'kalden/templates': {template_path}"
        )

    if target_path.exists() and target_path.is_dir():
        destination_path = target_path / template_path.name
    elif target_path.suffix:
        destination_path = target_path
    else:
        destination_path = target_path / template_path.name

    destination_path.parent.mkdir(parents=True, exist_ok=True)

    with as_file(template_resource) as source_path:
        copy2(source_path, destination_path)

    return destination_path
