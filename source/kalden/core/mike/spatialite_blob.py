"""Decode SpatiaLite geometry BLOBs without loading mod_spatialite.

This implements the classic BLOB-Geometry and TinyPoint layouts documented by
SpatiaLite.  It deliberately avoids sqlite3.load_extension(), which can terminate
an interpreter when an incompatible native DLL is loaded.
"""
from __future__ import annotations

import struct
from typing import Optional, Tuple

from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely import wkb, wkt


class SpatialiteBlobError(ValueError):
    pass


class _Reader:
    def __init__(self, data: bytes, endian: str, offset: int = 0):
        self.data = data
        self.endian = endian
        self.offset = offset

    def _unpack(self, fmt: str, size: int):
        end = self.offset + size
        if end > len(self.data):
            raise SpatialiteBlobError(
                "Unexpected end of geometry BLOB at byte {}.".format(self.offset)
            )
        value = struct.unpack_from(self.endian + fmt, self.data, self.offset)[0]
        self.offset = end
        return value

    def uint32(self) -> int:
        return int(self._unpack("I", 4))

    def float32(self) -> float:
        return float(self._unpack("f", 4))

    def float64(self) -> float:
        return float(self._unpack("d", 8))

    def byte(self) -> int:
        if self.offset >= len(self.data):
            raise SpatialiteBlobError(
                "Unexpected end of geometry BLOB at byte {}.".format(self.offset)
            )
        value = self.data[self.offset]
        self.offset += 1
        return value


def _type_info(type_code: int) -> Tuple[int, int, bool]:
    """Return base geometry type, dimension model, and compression flag."""
    if type_code >= 1_000_000:
        shifted = type_code - 1_000_000
        dimension = shifted // 1000
        base_type = shifted % 1000
        compressed = True
    else:
        dimension = type_code // 1000
        base_type = type_code % 1000
        compressed = False
    if base_type not in range(1, 8):
        raise SpatialiteBlobError("Unsupported SpatiaLite geometry class {}.".format(type_code))
    if dimension not in (0, 1, 2, 3):
        raise SpatialiteBlobError("Unsupported SpatiaLite dimension model {}.".format(type_code))
    if compressed and base_type not in (2, 3):
        raise SpatialiteBlobError("Compressed class {} is not supported by SpatiaLite.".format(type_code))
    return base_type, dimension, compressed


def _read_coordinate(reader: _Reader, dimension: int):
    x = reader.float64()
    y = reader.float64()
    if dimension == 0:  # XY
        return (x, y)
    third = reader.float64()
    if dimension == 1:  # XYZ
        return (x, y, third)
    if dimension == 2:  # XYM: Shapely 1.x/2.x portability; discard M.
        return (x, y)
    # XYZM: preserve Z and discard M.
    reader.float64()
    return (x, y, third)


def _read_compressed_coordinate(reader: _Reader, previous, dimension: int):
    x = previous[0] + reader.float32()
    y = previous[1] + reader.float32()
    if dimension == 0:
        return (x, y)
    if dimension == 1:
        z = previous[2] + reader.float32()
        return (x, y, z)
    if dimension == 2:
        reader.float64()  # M is stored as a double; discard it.
        return (x, y)
    z = previous[2] + reader.float32()
    reader.float64()  # M is stored as a double; discard it.
    return (x, y, z)


def _read_line_coordinates(reader: _Reader, dimension: int, compressed: bool):
    count = reader.uint32()
    if count == 0:
        return []
    first = _read_coordinate(reader, dimension)
    if not compressed:
        return [first] + [_read_coordinate(reader, dimension) for _ in range(count - 1)]
    if count == 1:
        return [first]
    coordinates = [first]
    for _ in range(max(0, count - 2)):
        coordinates.append(_read_compressed_coordinate(reader, coordinates[-1], dimension))
    coordinates.append(_read_coordinate(reader, dimension))
    return coordinates


def _empty_geometry(base_type: int):
    return {
        1: Point(),
        2: LineString(),
        3: Polygon(),
        4: MultiPoint([]),
        5: MultiLineString([]),
        6: MultiPolygon([]),
        7: GeometryCollection([]),
    }[base_type]


def _parse_entity(reader: _Reader):
    marker = reader.byte()
    if marker != 0x69:
        raise SpatialiteBlobError(
            "Invalid SpatiaLite collection marker 0x{:02X} at byte {}.".format(
                marker, reader.offset - 1
            )
        )
    entity_type = reader.uint32()
    return _parse_geometry_body(reader, entity_type)


def _parse_geometry_body(reader: _Reader, type_code: int):
    base_type, dimension, compressed = _type_info(type_code)

    if base_type == 1:
        if compressed:
            raise SpatialiteBlobError("A POINT cannot use compressed encoding.")
        return Point(_read_coordinate(reader, dimension))

    if base_type == 2:
        coordinates = _read_line_coordinates(reader, dimension, compressed)
        return LineString(coordinates) if coordinates else _empty_geometry(base_type)

    if base_type == 3:
        ring_count = reader.uint32()
        rings = [
            _read_line_coordinates(reader, dimension, compressed)
            for _ in range(ring_count)
        ]
        rings = [ring for ring in rings if ring]
        if not rings:
            return _empty_geometry(base_type)
        return Polygon(rings[0], rings[1:])

    entity_count = reader.uint32()
    entities = [_parse_entity(reader) for _ in range(entity_count)]
    if not entities:
        return _empty_geometry(base_type)
    if base_type == 4:
        points = [geom for geom in entities if geom.geom_type == "Point"]
        if len(points) != len(entities):
            raise SpatialiteBlobError("MULTIPOINT contains a non-POINT entity.")
        return MultiPoint(points)
    if base_type == 5:
        lines = [geom for geom in entities if geom.geom_type == "LineString"]
        if len(lines) != len(entities):
            raise SpatialiteBlobError("MULTILINESTRING contains a non-LINESTRING entity.")
        return MultiLineString(lines)
    if base_type == 6:
        polygons = [geom for geom in entities if geom.geom_type == "Polygon"]
        if len(polygons) != len(entities):
            raise SpatialiteBlobError("MULTIPOLYGON contains a non-POLYGON entity.")
        return MultiPolygon(polygons)
    return GeometryCollection(entities)


def decode_spatialite_blob(value) -> Tuple[object, Optional[int]]:
    """Decode one SpatiaLite BLOB and return ``(shapely_geometry, srid)``."""
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytearray):
        value = bytes(value)
    if not isinstance(value, bytes):
        raise SpatialiteBlobError("Geometry value is not a BLOB.")
    if len(value) < 8 or value[0] != 0x00 or value[-1] != 0xFE:
        raise SpatialiteBlobError("Value does not have a valid SpatiaLite BLOB envelope.")

    endian_byte = value[1]
    if endian_byte not in (0x00, 0x01, 0x80, 0x81):
        raise SpatialiteBlobError("Invalid SpatiaLite endian marker 0x{:02X}.".format(endian_byte))
    endian = "<" if (endian_byte & 0x01) else ">"
    srid = int(struct.unpack_from(endian + "i", value, 2)[0])

    if endian_byte & 0x80:  # TinyPoint
        tiny_type = value[6]
        dimensions = {1: 0, 2: 1, 3: 2, 4: 3}
        if tiny_type not in dimensions:
            raise SpatialiteBlobError("Unsupported TinyPoint type {}.".format(tiny_type))
        reader = _Reader(value, endian, 7)
        geometry = Point(_read_coordinate(reader, dimensions[tiny_type]))
        if reader.offset != len(value) - 1:
            raise SpatialiteBlobError("Unexpected trailing bytes in TinyPoint geometry.")
        return geometry, srid

    if len(value) < 44 or value[38] != 0x7C:
        raise SpatialiteBlobError("Invalid classic SpatiaLite BLOB header.")
    type_code = int(struct.unpack_from(endian + "I", value, 39)[0])
    reader = _Reader(value, endian, 43)
    geometry = _parse_geometry_body(reader, type_code)
    if reader.offset != len(value) - 1:
        raise SpatialiteBlobError(
            "Geometry decoder stopped at byte {}, but BLOB ends at byte {}.".format(
                reader.offset, len(value) - 1
            )
        )
    return geometry, srid


def decode_geometry_value(value) -> Tuple[object, Optional[int], str]:
    """Decode SpatiaLite BLOB, ordinary WKB, or WKT without native extensions."""
    if value is None:
        return None, None, "null"
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None, None, "null"
        return wkt.loads(text), None, "wkt"
    raw = value.tobytes() if isinstance(value, memoryview) else bytes(value)
    try:
        geometry, srid = decode_spatialite_blob(raw)
        return geometry, srid, "spatialite_blob"
    except SpatialiteBlobError:
        try:
            return wkb.loads(raw), None, "wkb"
        except Exception as exc:
            raise SpatialiteBlobError(
                "Geometry is neither a supported SpatiaLite BLOB nor ordinary WKB: {}".format(exc)
            )
